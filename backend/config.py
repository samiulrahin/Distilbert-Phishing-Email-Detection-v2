from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    model_dir: Path = Path(
        os.getenv(
            "PHISHING_MODEL_DIR",
            PROJECT_ROOT / "models" / "distilbert-phishing-v2",
        )
    )
    baseline_path: Path = Path(
        os.getenv(
            "PHISHING_BASELINE_PATH",
            PROJECT_ROOT / "models" / "tfidf-logistic-regression.joblib",
        )
    )
    log_path: Path = Path(
        os.getenv(
            "PHISHING_LOG_PATH",
            PROJECT_ROOT / "evaluation" / "results" / "runtime_log.csv",
        )
    )
    decision_threshold: float = float(os.getenv("PHISHING_THRESHOLD", "0.50"))
    max_input_chars: int = int(os.getenv("PHISHING_MAX_INPUT_CHARS", "50000"))
    allow_demo_fallback: bool = _as_bool(
        os.getenv("PHISHING_ALLOW_DEMO_FALLBACK"), True
    )
    enable_metrics_log: bool = _as_bool(
        os.getenv("PHISHING_ENABLE_METRICS_LOG"), True
    )


settings = Settings()
