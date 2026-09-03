from __future__ import annotations

import math

import numpy as np


def calibrate_probability(probability: float | np.ndarray, temperature: float) -> float | np.ndarray:
    values = np.asarray(probability, dtype=float)
    clipped = np.clip(values, 1e-7, 1.0 - 1e-7)
    logits = np.log(clipped / (1.0 - clipped)) / max(float(temperature), 1e-6)
    calibrated = 1.0 / (1.0 + np.exp(-logits))
    if np.isscalar(probability):
        return float(calibrated)
    return calibrated


def binary_log_loss(labels: np.ndarray, probability: np.ndarray) -> float:
    clipped = np.clip(probability, 1e-7, 1.0 - 1e-7)
    return float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)))


def fit_temperature(labels: np.ndarray, raw_probability: np.ndarray) -> tuple[float, dict[str, float]]:
    candidates = np.geomspace(0.25, 8.0, 600)
    losses = [
        binary_log_loss(labels, calibrate_probability(raw_probability, value))
        for value in candidates
    ]
    best_index = int(np.argmin(losses))
    temperature = float(candidates[best_index])
    return temperature, {
        "raw_log_loss": binary_log_loss(labels, raw_probability),
        "calibrated_log_loss": float(losses[best_index]),
    }


def triage_label(probability: float, legitimate_max: float, phishing_min: float) -> str:
    if probability <= legitimate_max:
        return "legitimate"
    if probability >= phishing_min:
        return "phishing"
    return "uncertain"


def _precision(labels: np.ndarray, mask: np.ndarray, expected: int) -> float:
    if not mask.any():
        return math.nan
    return float(np.mean(labels[mask] == expected))


def select_triage_policy(
    labels: np.ndarray,
    probability: np.ndarray,
    target_precision: float = 0.95,
    minimum_predictions: int = 4,
) -> tuple[float, float, dict[str, object]]:
    labels = np.asarray(labels, dtype=int)
    probability = np.asarray(probability, dtype=float)
    grid = np.linspace(0.001, 0.999, 999)

    legitimate_rows = []
    for threshold in grid[grid < 0.5]:
        mask = probability <= threshold
        count = int(mask.sum())
        precision = _precision(labels, mask, 0)
        if count >= minimum_predictions and precision >= target_precision:
            legitimate_rows.append((count, precision, float(threshold)))

    phishing_rows = []
    for threshold in grid[grid > 0.5]:
        mask = probability >= threshold
        count = int(mask.sum())
        precision = _precision(labels, mask, 1)
        if count >= minimum_predictions and precision >= target_precision:
            phishing_rows.append((count, precision, float(threshold)))

    if legitimate_rows:
        _, _, legitimate_max = max(legitimate_rows, key=lambda row: (row[0], row[1], row[2]))
    else:
        fallback = []
        for threshold in grid[grid < 0.5]:
            mask = probability <= threshold
            count = int(mask.sum())
            if count:
                fallback.append((_precision(labels, mask, 0), count, float(threshold)))
        _, _, legitimate_max = max(fallback, key=lambda row: (row[0], row[1], row[2]))

    if phishing_rows:
        _, _, phishing_min = max(phishing_rows, key=lambda row: (row[0], row[1], -row[2]))
    else:
        fallback = []
        for threshold in grid[grid > 0.5]:
            mask = probability >= threshold
            count = int(mask.sum())
            if count:
                fallback.append((_precision(labels, mask, 1), count, -float(threshold)))
        _, _, negative_threshold = max(fallback, key=lambda row: (row[0], row[1], row[2]))
        phishing_min = -negative_threshold

    predicted = np.asarray(
        [triage_label(value, legitimate_max, phishing_min) for value in probability]
    )
    decided = predicted != "uncertain"
    binary = np.where(predicted == "phishing", 1, 0)
    summary = {
        "target_precision": target_precision,
        "minimum_predictions_per_decided_class": minimum_predictions,
        "coverage": float(decided.mean()),
        "selective_accuracy": float(np.mean(binary[decided] == labels[decided])) if decided.any() else 0.0,
        "three_way_exact_accuracy": float(
            np.mean(
                ((predicted == "phishing") & (labels == 1))
                | ((predicted == "legitimate") & (labels == 0))
            )
        ),
        "legitimate_precision": _precision(labels, predicted == "legitimate", 0),
        "phishing_precision": _precision(labels, predicted == "phishing", 1),
        "legitimate_predictions": int(np.sum(predicted == "legitimate")),
        "phishing_predictions": int(np.sum(predicted == "phishing")),
        "uncertain_predictions": int(np.sum(predicted == "uncertain")),
    }
    return round(legitimate_max, 6), round(phishing_min, 6), summary


def triage_metrics(
    labels: np.ndarray,
    probability: np.ndarray,
    legitimate_max: float,
    phishing_min: float,
) -> dict[str, object]:
    labels = np.asarray(labels, dtype=int)
    predicted = np.asarray(
        [triage_label(value, legitimate_max, phishing_min) for value in probability]
    )
    decided = predicted != "uncertain"
    binary = np.where(predicted == "phishing", 1, 0)
    false_positive = int(np.sum((predicted == "phishing") & (labels == 0)))
    false_negative = int(np.sum((predicted == "legitimate") & (labels == 1)))
    return {
        "rows": int(len(labels)),
        "coverage": float(decided.mean()),
        "selective_accuracy": float(np.mean(binary[decided] == labels[decided])) if decided.any() else 0.0,
        "three_way_exact_accuracy": float(np.mean(((predicted == "phishing") & (labels == 1)) | ((predicted == "legitimate") & (labels == 0)))),
        "legitimate_precision": _precision(labels, predicted == "legitimate", 0),
        "phishing_precision": _precision(labels, predicted == "phishing", 1),
        "legitimate_predictions": int(np.sum(predicted == "legitimate")),
        "phishing_predictions": int(np.sum(predicted == "phishing")),
        "uncertain_predictions": int(np.sum(predicted == "uncertain")),
        "false_positive": false_positive,
        "false_negative": false_negative,
    }
