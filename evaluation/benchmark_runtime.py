from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import psutil
import requests
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def percentile(values: list[float], percentile_value: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile_value)))
    return ordered[index]


def wait_for_server(url: str, timeout_seconds: float = 180.0) -> tuple[dict[str, object], float]:
    start = time.perf_counter()
    while time.perf_counter() - start < timeout_seconds:
        try:
            response = requests.get(f"{url}/health", timeout=2)
            if response.ok:
                return response.json(), (time.perf_counter() - start) * 1000
        except requests.RequestException:
            time.sleep(0.25)
    raise TimeoutError("Local API did not become ready before the timeout.")


def benchmark(args: argparse.Namespace) -> dict[str, object]:
    emails = json.loads(args.emails.read_text(encoding="utf-8"))
    process = None
    server_process = None
    if args.start_server:
        server_environment = os.environ.copy()
        if args.force_baseline:
            server_environment["PHISHING_MODEL_DIR"] = str(
                PROJECT_ROOT / "models" / "disabled-for-baseline-benchmark"
            )
        server_process = subprocess.Popen(
            [sys.executable, "-m", "backend"],
            cwd=PROJECT_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=server_environment,
        )
        process = psutil.Process(server_process.pid)

    rows: list[dict[str, object]] = []
    try:
        health, startup_ms = wait_for_server(args.url)
        if process:
            candidates = [process, *process.children(recursive=True)]
            live_candidates = [candidate for candidate in candidates if candidate.is_running()]
            process = max(live_candidates, key=lambda candidate: candidate.memory_info().rss)
        cold_payload = {
            "subject": emails[0]["subject"],
            "body": emails[0]["body"],
            "source": "api-test",
        }
        cold_start = time.perf_counter()
        cold_response = requests.post(
            f"{args.url}/predict", json=cold_payload, timeout=120
        )
        cold_end_to_end_ms = (time.perf_counter() - cold_start) * 1000
        cold_response.raise_for_status()
        cold_result = cold_response.json()
        for _ in range(args.warmup):
            requests.post(
                f"{args.url}/predict",
                json={**emails[0], "source": "api-test"},
                timeout=60,
            ).raise_for_status()

        for run_index in range(args.runs):
            email = emails[run_index % len(emails)]
            payload = {
                "subject": email["subject"],
                "body": email["body"],
                "source": "api-test",
            }
            rss_before = process.memory_info().rss / (1024 * 1024) if process else None
            cpu_before = process.cpu_times() if process else None
            start = time.perf_counter()
            response = requests.post(f"{args.url}/predict", json=payload, timeout=60)
            end_to_end_ms = (time.perf_counter() - start) * 1000
            response.raise_for_status()
            result = response.json()
            cpu_ms = None
            rss_after = None
            if process and cpu_before:
                cpu_after = process.cpu_times()
                cpu_ms = ((cpu_after.user + cpu_after.system) - (cpu_before.user + cpu_before.system)) * 1000
                rss_after = process.memory_info().rss / (1024 * 1024)
            rows.append(
                {
                    "run": run_index + 1,
                    "email_id": email["id"],
                    "expected_label": email["expected_label"],
                    "predicted_label": result["label"],
                    "model_mode": result["model_mode"],
                    "model_inference_ms": result["inference_ms"],
                    "http_end_to_end_ms": round(end_to_end_ms, 3),
                    "backend_cpu_ms": round(cpu_ms, 3) if cpu_ms is not None else "",
                    "backend_rss_before_mb": round(rss_before, 3) if rss_before is not None else "",
                    "backend_rss_after_mb": round(rss_after, 3) if rss_after is not None else "",
                }
            )
    finally:
        if server_process:
            server_process.terminate()
            try:
                server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_process.kill()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or ("baseline_runtime" if args.force_baseline else "runtime")
    csv_path = args.results_dir / f"{prefix}_benchmark_runs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    inference_values = [float(row["model_inference_ms"]) for row in rows]
    end_to_end_values = [float(row["http_end_to_end_ms"]) for row in rows]
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    axis.boxplot(
        [inference_values, end_to_end_values],
        tick_labels=["Model inference", "HTTP end-to-end"],
        patch_artist=True,
        boxprops={"facecolor": "#b9e1e2", "edgecolor": "#087f83"},
        medianprops={"color": "#b42318", "linewidth": 2},
    )
    axis.set_ylabel("Milliseconds")
    axis.set_title("Warmed local prediction latency")
    axis.grid(axis="y", alpha=0.24)
    figure.tight_layout()
    figure.savefig(args.results_dir / f"{prefix}_latency_distribution.png", dpi=180)
    plt.close(figure)

    end_to_end = end_to_end_values
    inference = inference_values
    summary = {
        "model_name": health["model_name"],
        "model_mode": health["model_mode"],
        "model_ready": health["model_ready"],
        "startup_ms": round(startup_ms, 3),
        "cold_first_prediction": {
            "model_inference_ms": cold_result["inference_ms"],
            "http_end_to_end_ms": round(cold_end_to_end_ms, 3),
        },
        "warmup_runs": args.warmup,
        "measured_runs": len(rows),
        "model_inference_ms": {
            "mean": round(statistics.fmean(inference), 3),
            "median": round(statistics.median(inference), 3),
            "p95": round(percentile(inference, 0.95), 3),
            "min": round(min(inference), 3),
            "max": round(max(inference), 3),
        },
        "http_end_to_end_ms": {
            "mean": round(statistics.fmean(end_to_end), 3),
            "median": round(statistics.median(end_to_end), 3),
            "p95": round(percentile(end_to_end, 0.95), 3),
            "min": round(min(end_to_end), 3),
            "max": round(max(end_to_end), 3),
        },
        "backend_rss_mb": {
            "mean": round(statistics.fmean(float(row["backend_rss_after_mb"]) for row in rows), 3)
            if process else None,
            "peak": round(max(float(row["backend_rss_after_mb"]) for row in rows), 3)
            if process else None,
        },
        "controlled_accuracy": round(
            sum(row["expected_label"] == row["predicted_label"] for row in rows) / len(rows), 4
        ),
        "decision_counts": {
            label: sum(row["predicted_label"] == label for row in rows)
            for label in ("legitimate", "phishing", "uncertain")
        },
        "note": "Controlled synthetic workflow emails; not a substitute for held-out public-dataset evaluation.",
    }
    (args.results_dir / f"{prefix}_benchmark_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark local model and HTTP workflow performance.")
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--emails", type=Path, default=PROJECT_ROOT / "evaluation" / "controlled_test_emails" / "controlled_emails.json")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "evaluation" / "results")
    parser.add_argument("--runs", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--prefix", help="Output filename prefix, for example v2_runtime.")
    parser.add_argument("--start-server", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--force-baseline",
        action="store_true",
        help="Disable the transformer checkpoint so the classical baseline is measured.",
    )
    return parser


if __name__ == "__main__":
    print(json.dumps(benchmark(build_parser().parse_args()), indent=2))
