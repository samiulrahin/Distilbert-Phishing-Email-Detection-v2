from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from ml.metrics import (
    binary_metrics,
    save_confusion_matrix,
    save_metrics,
    select_f1_threshold,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dataframe_dataset(
    path: Path,
    max_samples: int | None,
    seed: int,
    augmentation_path: Path | None = None,
) -> Dataset:
    frame = pd.read_csv(path)
    if augmentation_path:
        augmentation = pd.read_csv(augmentation_path)
        required = {"text", "label", "example_id"}
        if not required.issubset(augmentation.columns):
            raise ValueError(f"Augmentation data must contain: {sorted(required)}")
        frame = pd.concat([frame, augmentation[list(frame.columns)]], ignore_index=True)
        frame = frame.drop_duplicates(subset=["example_id"], keep="first")
    if max_samples and len(frame) > max_samples:
        groups = []
        for _, group in frame.groupby("label"):
            sample_count = max(1, round(max_samples * len(group) / len(frame)))
            groups.append(group.sample(n=min(sample_count, len(group)), random_state=seed))
        frame = pd.concat(groups, ignore_index=True)
        if len(frame) > max_samples:
            frame = frame.sample(max_samples, random_state=seed)
    return Dataset.from_pandas(frame[["text", "label", "example_id"]], preserve_index=False)


def train(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=2,
        id2label={0: "legitimate", 1: "phishing"},
        label2id={"legitimate": 0, "phishing": 1},
    )

    original_train_rows = len(pd.read_csv(args.train))
    augmentation_source_rows = len(pd.read_csv(args.augmentation)) if args.augmentation else 0
    train_dataset = dataframe_dataset(
        args.train, args.max_train_samples, args.seed, args.augmentation
    )
    validation_dataset = dataframe_dataset(args.validation, args.max_validation_samples, args.seed)
    test_dataset = dataframe_dataset(args.test, args.max_test_samples, args.seed)

    def tokenize(batch: dict[str, list[object]]) -> dict[str, object]:
        return tokenizer(
            batch["text"], truncation=True, max_length=args.max_length, padding=False
        )

    train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["text", "example_id"])
    validation_labels = np.asarray(validation_dataset["label"])
    validation_dataset = validation_dataset.map(tokenize, batched=True, remove_columns=["text", "example_id"])
    test_example_ids = list(test_dataset["example_id"])
    test_labels = np.asarray(test_dataset["label"])
    test_dataset = test_dataset.map(tokenize, batched=True, remove_columns=["text", "example_id"])

    def compute_metrics(prediction) -> dict[str, float]:
        logits, labels = prediction
        probability = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
        return binary_metrics(np.asarray(labels), probability)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=0.1,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    train_result = trainer.train()
    validation_output = trainer.predict(validation_dataset)
    validation_probability = torch.softmax(
        torch.tensor(validation_output.predictions), dim=-1
    ).numpy()[:, 1]
    threshold, threshold_analysis = select_f1_threshold(
        validation_labels, validation_probability
    )
    test_output = trainer.predict(test_dataset)
    probability = torch.softmax(torch.tensor(test_output.predictions), dim=-1).numpy()[:, 1]
    metrics: dict[str, object] = binary_metrics(test_labels, probability, threshold)
    metrics.update(
        {
            "model": args.base_model,
            "split": "held-out-test",
            "train_rows": len(train_dataset),
            "validation_rows": len(validation_dataset),
            "test_rows": len(test_dataset),
            "epochs_requested": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_length": args.max_length,
            "random_seed": args.seed,
            "augmentation": str(args.augmentation) if args.augmentation else None,
            "augmentation_source_rows": augmentation_source_rows,
            "augmentation_effective_unique_rows": max(0, len(train_dataset) - original_train_rows),
            "train_runtime_seconds": round(train_result.metrics.get("train_runtime", 0.0), 3),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    (args.output_dir / "decision_threshold.json").write_text(
        json.dumps({"threshold": threshold, "selected_on": "validation", "criterion": "maximum_f1"}, indent=2),
        encoding="utf-8",
    )
    args.results_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.result_prefix
    save_metrics(metrics, args.results_dir / f"{prefix}_metrics.json")
    pd.DataFrame(threshold_analysis).to_csv(
        args.results_dir / f"{prefix}_threshold_analysis.csv", index=False
    )
    save_confusion_matrix(
        test_labels,
        probability,
        args.results_dir / f"{prefix}_confusion_matrix.png",
        threshold,
    )
    pd.DataFrame(
        {
            "true_label": validation_labels,
            "raw_phishing_probability": np.round(validation_probability, 8),
        }
    ).to_csv(args.results_dir / f"{prefix}_validation_predictions.csv", index=False)
    pd.DataFrame(
        {
            "example_id": test_example_ids,
            "true_label": test_labels,
            "phishing_probability": np.round(probability, 6),
            "predicted_label": (probability >= threshold).astype(int),
        }
    ).to_csv(args.results_dir / f"{prefix}_test_predictions.csv", index=False)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for phishing email detection.")
    parser.add_argument("--train", type=Path, default=PROJECT_ROOT / "data" / "splits" / "train.csv")
    parser.add_argument("--validation", type=Path, default=PROJECT_ROOT / "data" / "splits" / "validation.csv")
    parser.add_argument("--test", type=Path, default=PROJECT_ROOT / "data" / "splits" / "test.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "models" / "distilbert-phishing-v1")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "evaluation" / "results")
    parser.add_argument("--base-model", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augmentation", type=Path)
    parser.add_argument("--result-prefix", default="distilbert")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    print(json.dumps(train(arguments), indent=2))
