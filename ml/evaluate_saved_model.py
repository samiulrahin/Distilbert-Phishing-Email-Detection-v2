from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ml.metrics import (
    binary_metrics,
    save_confusion_matrix,
    save_metrics,
    select_f1_threshold,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def probabilities(
    texts: list[str], tokenizer, model, device: torch.device, batch_size: int = 32
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            truncation=True,
            max_length=256,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded).logits
            outputs.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(outputs)


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)
    validation = pd.read_csv(args.validation)
    test = pd.read_csv(args.test)
    validation_probability = probabilities(
        validation["text"].tolist(), tokenizer, model, device, args.batch_size
    )
    threshold, analysis = select_f1_threshold(
        validation["label"].to_numpy(), validation_probability
    )
    test_probability = probabilities(
        test["text"].tolist(), tokenizer, model, device, args.batch_size
    )
    metrics_path = args.results_dir / "distilbert_metrics.json"
    previous = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    metrics: dict[str, object] = {
        **previous,
        **binary_metrics(test["label"].to_numpy(), test_probability, threshold),
        "model": previous.get("model", "distilbert-base-uncased"),
        "split": "held-out-test",
        "validation_rows": len(validation),
        "test_rows": len(test),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }
    save_metrics(metrics, metrics_path)
    pd.DataFrame(analysis).to_csv(
        args.results_dir / "distilbert_threshold_analysis.csv", index=False
    )
    save_confusion_matrix(
        test["label"].to_numpy(),
        test_probability,
        args.results_dir / "distilbert_confusion_matrix.png",
        threshold,
    )
    pd.DataFrame(
        {
            "example_id": test["example_id"],
            "true_label": test["label"],
            "phishing_probability": np.round(test_probability, 6),
            "predicted_label": (test_probability >= threshold).astype(int),
        }
    ).to_csv(args.results_dir / "distilbert_test_predictions.csv", index=False)
    (args.model_dir / "decision_threshold.json").write_text(
        json.dumps(
            {
                "threshold": threshold,
                "selected_on": "validation",
                "criterion": "maximum_f1_with_nearest_0.5_tie_break",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Re-evaluate an existing saved DistilBERT checkpoint.")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models" / "distilbert-phishing-v1")
    parser.add_argument("--validation", type=Path, default=PROJECT_ROOT / "data" / "splits" / "validation.csv")
    parser.add_argument("--test", type=Path, default=PROJECT_ROOT / "data" / "splits" / "test.csv")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "evaluation" / "results")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser


if __name__ == "__main__":
    print(json.dumps(evaluate(build_parser().parse_args()), indent=2))
