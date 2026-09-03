from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def binary_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    predicted = (probability >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, predicted, average="binary", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "threshold": float(threshold),
    }


def select_f1_threshold(
    y_true: np.ndarray, probability: np.ndarray
) -> tuple[float, list[dict[str, float]]]:
    analysis: list[dict[str, float]] = []
    for threshold in np.linspace(0.10, 0.90, 81):
        metrics = binary_metrics(y_true, probability, float(threshold))
        analysis.append(metrics)
    best = max(
        analysis,
        key=lambda row: (
            row["f1"],
            row["recall"],
            row["precision"],
            -abs(row["threshold"] - 0.5),
        ),
    )
    return float(best["threshold"]), analysis


def save_metrics(metrics: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def save_confusion_matrix(y_true: np.ndarray, probability: np.ndarray, path: Path, threshold: float = 0.5) -> None:
    matrix = confusion_matrix(y_true, (probability >= threshold).astype(int), labels=[0, 1])
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.2, 5.0))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Legitimate", "Phishing"],
        yticklabels=["Legitimate", "Phishing"],
        ax=axis,
    )
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title("Held-out test confusion matrix")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
