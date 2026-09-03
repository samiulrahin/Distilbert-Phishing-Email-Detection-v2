from __future__ import annotations

from dataclasses import replace

from fastapi.testclient import TestClient

from backend.app import create_app
from backend.config import settings


def make_test_settings(tmp_path):
    return replace(
        settings,
        model_dir=tmp_path / "missing-transformer",
        baseline_path=tmp_path / "missing-baseline.joblib",
        log_path=tmp_path / "metrics.csv",
        enable_metrics_log=True,
        allow_demo_fallback=True,
    )


def test_health_reports_active_model_mode(tmp_path):
    client = TestClient(create_app(make_test_settings(tmp_path)))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_name": "controlled-rule-demonstrator",
        "model_version": "demo-v1",
        "model_mode": "demonstration",
        "model_ready": False,
        "policy_mode": "legacy-threshold",
        "privacy": "local-only",
    }


def test_demo_interface_is_served(tmp_path):
    client = TestClient(create_app(make_test_settings(tmp_path)))
    response = client.get("/demo")
    assert response.status_code == 200
    assert "Mail Risk Lab" in response.text
    assert "Analyse an email locally" in response.text


def test_predict_contract_and_privacy_label(tmp_path):
    client = TestClient(create_app(make_test_settings(tmp_path)))
    response = client.post(
        "/predict",
        json={
            "subject": "Urgent account verification",
            "body": "Your account will be locked. Click https://example.invalid and enter your password.",
            "source": "api-test",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["label"] == "phishing"
    assert payload["privacy"] == "local-only"
    assert payload["model_mode"] == "demonstration"
    assert payload["confidence"] >= 0.5
    assert payload["legitimate_max"] < payload["phishing_min"]
    assert payload["policy_mode"] == "legacy-threshold"
    assert payload["inference_ms"] >= 0
    assert payload["risk_signals"]


def test_empty_email_is_rejected(tmp_path):
    client = TestClient(create_app(make_test_settings(tmp_path)))
    response = client.post(
        "/predict", json={"subject": "  ", "body": "", "source": "api-test"}
    )
    assert response.status_code == 422


def test_unknown_source_is_rejected(tmp_path):
    client = TestClient(create_app(make_test_settings(tmp_path)))
    response = client.post(
        "/predict", json={"subject": "Hello", "body": "World", "source": "external"}
    )
    assert response.status_code == 422
