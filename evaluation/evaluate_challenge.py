from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backend.model_loader import ModelService
from ml.policy import (
    calibrate_probability,
    fit_temperature,
    select_triage_policy,
    triage_label,
    triage_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_records(path: Path) -> list[dict[str, object]]:
    return json.loads(path.read_text(encoding="utf-8"))


def score_records(service: ModelService, records: list[dict[str, object]]) -> np.ndarray:
    return np.asarray(
        [service.predict_raw(str(row["subject"]), str(row["body"])) for row in records],
        dtype=float,
    )


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    validation = pd.read_csv(args.validation_predictions)
    validation_labels = validation["true_label"].to_numpy(dtype=int)
    validation_raw = validation["raw_phishing_probability"].to_numpy(dtype=float)
    temperature, calibration = fit_temperature(validation_labels, validation_raw)

    service = ModelService(
        args.model_dir,
        PROJECT_ROOT / "models" / "missing-baseline.joblib",
        allow_demo_fallback=False,
    )
    development = load_records(args.development)
    test = load_records(args.test)
    development_raw = score_records(service, development)
    test_raw = score_records(service, test)
    development_probability = calibrate_probability(development_raw, temperature)
    test_probability = calibrate_probability(test_raw, temperature)
    development_labels = np.asarray([row["label"] for row in development], dtype=int)
    test_labels = np.asarray([row["label"] for row in test], dtype=int)

    legitimate_max, phishing_min, selection = select_triage_policy(
        development_labels,
        development_probability,
        target_precision=args.target_precision,
        minimum_predictions=args.minimum_predictions,
    )
    policy = {
        "version": "selective-v2",
        "temperature": round(temperature, 6),
        "legitimate_max": legitimate_max,
        "phishing_min": phishing_min,
        "calibration_source": "original validation split",
        "policy_selection_source": "controlled development challenge",
        "target_decided_class_precision": args.target_precision,
        "minimum_predictions_per_decided_class": args.minimum_predictions,
        "calibration": calibration,
        "development_selection": selection,
        "interpretation": "Scores between legitimate_max and phishing_min are uncertain and require manual review.",
    }
    (args.model_dir / "decision_policy.json").write_text(
        json.dumps(policy, indent=2), encoding="utf-8"
    )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    output_rows = []
    for split, records, raw, calibrated in (
        ("development", development, development_raw, development_probability),
        ("test", test, test_raw, test_probability),
    ):
        for row, raw_value, calibrated_value in zip(records, raw, calibrated):
            output_rows.append(
                {
                    "id": row["id"],
                    "split": split,
                    "scenario": row["scenario"],
                    "expected_label": row["expected_label"],
                    "raw_model_score": round(float(raw_value), 6),
                    "calibrated_risk_score": round(float(calibrated_value), 6),
                    "predicted_label": triage_label(float(calibrated_value), legitimate_max, phishing_min),
                }
            )
    pd.DataFrame(output_rows).to_csv(
        args.results_dir / "v2_challenge_predictions.csv", index=False
    )
    results = {
        "model_name": service.model_name,
        "model_version": service.model_version,
        "temperature": round(temperature, 6),
        "legitimate_max": legitimate_max,
        "phishing_min": phishing_min,
        "calibration": calibration,
        "development": triage_metrics(
            development_labels, development_probability, legitimate_max, phishing_min
        ),
        "untouched_test": triage_metrics(
            test_labels, test_probability, legitimate_max, phishing_min
        ),
        "limitations": [
            "The challenge set is controlled and synthetic rather than prevalence-representative inbox traffic.",
            "Policy thresholds were selected on the development challenge and evaluated once on the separate test challenge.",
            "Content-only analysis cannot authenticate the sender or destination domain.",
        ],
    }
    (args.results_dir / "v2_challenge_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate and evaluate the Version 2 challenge policy.")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models" / "distilbert-phishing-v2")
    parser.add_argument("--validation-predictions", type=Path, default=PROJECT_ROOT / "evaluation" / "results" / "distilbert_v2_validation_predictions.csv")
    parser.add_argument("--development", type=Path, default=PROJECT_ROOT / "evaluation" / "challenge_emails" / "development.json")
    parser.add_argument("--test", type=Path, default=PROJECT_ROOT / "evaluation" / "challenge_emails" / "test.json")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "evaluation" / "results")
    parser.add_argument("--target-precision", type=float, default=0.95)
    parser.add_argument("--minimum-predictions", type=int, default=4)
    return parser


if __name__ == "__main__":
    print(json.dumps(evaluate(build_parser().parse_args()), indent=2))
