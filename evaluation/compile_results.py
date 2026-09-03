from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"


def compile_results(results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    sources = [
        ("TF-IDF + Logistic Regression", results_dir / "baseline_metrics.json"),
        ("DistilBERT V1", results_dir / "distilbert_metrics.json"),
        ("DistilBERT V2", results_dir / "distilbert_v2_metrics.json"),
    ]
    rows = []
    for display_name, path in sources:
        if not path.exists():
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "model": display_name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "threshold": metrics["threshold"],
                "test_rows": metrics["test_rows"],
            }
        )
    if not rows:
        raise FileNotFoundError("No model metric JSON files are available to compile.")

    frame = pd.DataFrame(rows)
    frame.to_csv(results_dir / "metrics_model.csv", index=False)
    chart = frame.set_index("model")[["accuracy", "precision", "recall", "f1", "roc_auc"]].T
    chart.index = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    axis = chart.plot(
        kind="bar",
        figsize=(9.2, 5.4),
        width=0.72,
        color=["#56636f", "#b66a00", "#087f83"][: len(frame)],
    )
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Score")
    axis.set_xlabel("")
    axis.set_title("Held-out test performance by model")
    axis.tick_params(axis="x", rotation=0)
    axis.legend(title="Model", loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    for container in axis.containers:
        axis.bar_label(container, fmt="%.3f", padding=3, fontsize=8)
    axis.grid(axis="y", alpha=0.24)
    figure = axis.get_figure()
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    figure.savefig(results_dir / "model_comparison.png", dpi=180)
    plt.close(figure)

    markdown = [
        "# Generated Model Evaluation Summary",
        "",
        "All values below come from the untouched held-out test split. Each threshold was selected on validation data by maximum F1 before test evaluation.",
        "",
        frame.to_markdown(index=False, floatfmt=".4f"),
        "",
        "These results describe this dataset split and experiment configuration; they are not production performance guarantees.",
    ]
    (results_dir / "model_evaluation_summary.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    compile_runtime(results_dir)
    return frame


def compile_runtime(results_dir: Path) -> pd.DataFrame | None:
    sources = [
        ("TF-IDF + Logistic Regression", results_dir / "baseline_runtime_benchmark_summary.json"),
        ("DistilBERT V1", results_dir / "runtime_benchmark_summary.json"),
        ("DistilBERT V2", results_dir / "v2_runtime_benchmark_summary.json"),
    ]
    rows = []
    for display_name, path in sources:
        if not path.exists():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "model": display_name,
                "startup_ms": result["startup_ms"],
                "cold_first_end_to_end_ms": result.get("cold_first_prediction", {}).get(
                    "http_end_to_end_ms"
                ),
                "inference_mean_ms": result["model_inference_ms"]["mean"],
                "inference_p95_ms": result["model_inference_ms"]["p95"],
                "end_to_end_mean_ms": result["http_end_to_end_ms"]["mean"],
                "end_to_end_p95_ms": result["http_end_to_end_ms"]["p95"],
                "rss_mean_mb": result["backend_rss_mb"]["mean"],
                "rss_peak_mb": result["backend_rss_mb"]["peak"],
                "controlled_accuracy": result["controlled_accuracy"],
            }
        )
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    frame.to_csv(results_dir / "metrics_runtime.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.8))
    frame.set_index("model")[["inference_p95_ms", "end_to_end_p95_ms"]].plot(
        kind="bar", ax=axes[0], color=["#087f83", "#b66a00"], rot=0
    )
    axes[0].set_title("Warmed p95 latency")
    axes[0].set_ylabel("Milliseconds")
    axes[0].set_xlabel("")
    axes[0].legend(["Model inference", "HTTP end-to-end"], loc="upper left")
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%.1f", padding=3, fontsize=8)
    axes[0].grid(axis="y", alpha=0.24)

    frame.set_index("model")[["rss_mean_mb"]].plot(
        kind="bar", ax=axes[1], color=["#56636f"], legend=False, rot=0
    )
    axes[1].set_title("Backend memory footprint")
    axes[1].set_ylabel("Mean resident memory (MB)")
    axes[1].set_xlabel("")
    axes[1].grid(axis="y", alpha=0.24)
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.0f", padding=3, fontsize=8)
    figure.tight_layout()
    figure.savefig(results_dir / "runtime_model_comparison.png", dpi=180)
    plt.close(figure)
    return frame


if __name__ == "__main__":
    print(compile_results().to_string(index=False))
