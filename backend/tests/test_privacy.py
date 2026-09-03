from __future__ import annotations

from backend.privacy import PrivacySafeMetricsLogger


def test_metrics_log_contains_no_raw_email_text(tmp_path):
    path = tmp_path / "runtime.csv"
    secret_subject = "private payroll subject"
    secret_body = "private message body and account number"
    logger = PrivacySafeMetricsLogger(path)
    logger.record(
        source="api-test",
        subject_chars=len(secret_subject),
        body_chars=len(secret_body),
        model_mode="demonstration",
        label="phishing",
        confidence=0.9,
        inference_ms=1.2,
    )
    content = path.read_text(encoding="utf-8")
    assert secret_subject not in content
    assert secret_body not in content
    assert "subject_chars" in content
    assert "body_chars" in content

