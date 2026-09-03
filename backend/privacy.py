from __future__ import annotations

import csv
import threading
from datetime import UTC, datetime
from pathlib import Path


LOG_FIELDS = (
    "timestamp_utc",
    "source",
    "subject_chars",
    "body_chars",
    "model_mode",
    "label",
    "confidence",
    "inference_ms",
)


class PrivacySafeMetricsLogger:
    """Records performance metadata only; raw email text never reaches disk."""

    def __init__(self, path: Path, enabled: bool = True) -> None:
        self.path = path
        self.enabled = enabled
        self._lock = threading.Lock()

    def record(
        self,
        *,
        source: str,
        subject_chars: int,
        body_chars: int,
        model_mode: str,
        label: str,
        confidence: float,
        inference_ms: float,
    ) -> None:
        if not self.enabled:
            return

        row = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "source": source,
            "subject_chars": subject_chars,
            "body_chars": body_chars,
            "model_mode": model_mode,
            "label": label,
            "confidence": f"{confidence:.6f}",
            "inference_ms": f"{inference_ms:.3f}",
        }
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            new_file = not self.path.exists()
            with self.path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
                if new_file:
                    writer.writeheader()
                writer.writerow(row)

