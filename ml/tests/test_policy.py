from __future__ import annotations

import numpy as np

from ml.policy import calibrate_probability, select_triage_policy, triage_label, triage_metrics


def test_temperature_calibration_preserves_order_and_softens_scores():
    raw = np.asarray([0.01, 0.25, 0.75, 0.99])
    calibrated = calibrate_probability(raw, 2.0)
    assert np.all(np.diff(calibrated) > 0)
    assert calibrated[0] > raw[0]
    assert calibrated[-1] < raw[-1]


def test_triage_label_has_review_band():
    assert triage_label(0.10, 0.20, 0.80) == "legitimate"
    assert triage_label(0.50, 0.20, 0.80) == "uncertain"
    assert triage_label(0.90, 0.20, 0.80) == "phishing"


def test_policy_selection_meets_precision_on_separable_development_data():
    labels = np.asarray([0] * 6 + [1] * 6)
    probability = np.asarray([0.01, 0.03, 0.08, 0.15, 0.45, 0.70, 0.30, 0.60, 0.82, 0.91, 0.96, 0.99])
    low, high, summary = select_triage_policy(labels, probability, target_precision=0.95, minimum_predictions=3)
    metrics = triage_metrics(labels, probability, low, high)
    assert summary["legitimate_precision"] >= 0.95
    assert summary["phishing_precision"] >= 0.95
    assert metrics["uncertain_predictions"] >= 1
