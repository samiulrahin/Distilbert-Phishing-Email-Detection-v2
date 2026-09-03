from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ml.metrics import (
    binary_metrics,
    save_confusion_matrix,
    save_metrics,
    select_f1_threshold,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def train(
    train_path: Path,
    validation_path: Path,
    test_path: Path,
    model_path: Path,
    results_dir: Path,
) -> dict[str, object]:
    train_frame = pd.read_csv(train_path)
    validation_frame = pd.read_csv(validation_path)
    test_frame = pd.read_csv(test_path)
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.995,
                    max_features=100_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1500,
                    class_weight="balanced",
                    random_state=42,
                    solver="liblinear",
                ),
            ),
        ]
    )
    start = perf_counter()
    pipeline.fit(train_frame["text"], train_frame["label"])
    training_seconds = perf_counter() - start
    validation_probability = pipeline.predict_proba(validation_frame["text"])[:, 1]
    threshold, threshold_analysis = select_f1_threshold(
        validation_frame["label"].to_numpy(), validation_probability
    )
    probability = pipeline.predict_proba(test_frame["text"])[:, 1]
    metrics: dict[str, object] = binary_metrics(
        test_frame["label"].to_numpy(), probability, threshold
    )
    metrics.update(
        {
            "model": "tfidf-logistic-regression",
            "split": "held-out-test",
            "train_rows": len(train_frame),
            "validation_rows": len(validation_frame),
            "test_rows": len(test_frame),
            "training_seconds": round(training_seconds, 3),
            "random_seed": 42,
        }
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    (model_path.parent / "baseline_threshold.json").write_text(
        json.dumps({"threshold": threshold, "selected_on": "validation", "criterion": "maximum_f1"}, indent=2),
        encoding="utf-8",
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    save_metrics(metrics, results_dir / "baseline_metrics.json")
    pd.DataFrame(threshold_analysis).to_csv(
        results_dir / "baseline_threshold_analysis.csv", index=False
    )
    save_confusion_matrix(
        test_frame["label"].to_numpy(),
        probability,
        results_dir / "baseline_confusion_matrix.png",
        threshold,
    )
    predictions = pd.DataFrame(
        {
            "example_id": test_frame["example_id"],
            "true_label": test_frame["label"],
            "phishing_probability": np.round(probability, 6),
            "predicted_label": (probability >= threshold).astype(int),
        }
    )
    predictions.to_csv(results_dir / "baseline_test_predictions.csv", index=False)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TF-IDF logistic regression baseline.")
    parser.add_argument("--train", type=Path, default=PROJECT_ROOT / "data" / "splits" / "train.csv")
    parser.add_argument("--validation", type=Path, default=PROJECT_ROOT / "data" / "splits" / "validation.csv")
    parser.add_argument("--test", type=Path, default=PROJECT_ROOT / "data" / "splits" / "test.csv")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models" / "tfidf-logistic-regression.joblib")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "evaluation" / "results")
    args = parser.parse_args()
    print(
        json.dumps(
            train(args.train, args.validation, args.test, args.model, args.results_dir),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
