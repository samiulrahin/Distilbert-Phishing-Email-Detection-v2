from __future__ import annotations

import re


SIGNAL_PATTERNS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    (
        "credential-request",
        "Requests account credentials or identity verification.",
        re.compile(r"\b(password|passcode|login|sign[ -]?in|verify (?:your )?account|credentials?)\b", re.I),
    ),
    (
        "urgency",
        "Uses urgent or time-pressure language.",
        re.compile(r"\b(urgent|immediately|within 24 hours|final warning|act now|suspended|expires? today)\b", re.I),
    ),
    (
        "payment",
        "Refers to payment, refunds, invoices, or financial details.",
        re.compile(r"\b(payment|invoice|refund|bank|card details?|wire transfer|crypto|gift card)\b", re.I),
    ),
    (
        "link-action",
        "Contains a link or asks the reader to click.",
        re.compile(r"(?:https?://|www\.|\bclick (?:here|the link)\b)", re.I),
    ),
    (
        "threat-or-consequence",
        "Threatens account or service consequences.",
        re.compile(r"\b(account (?:will be )?(?:closed|locked|disabled)|legal action|penalty|unauthori[sz]ed access)\b", re.I),
    ),
)


def identify_risk_signals(text: str, limit: int = 3) -> list[dict[str, str]]:
    signals: list[dict[str, str]] = []
    for category, description, pattern in SIGNAL_PATTERNS:
        if pattern.search(text):
            signals.append({"category": category, "description": description})
        if len(signals) == limit:
            break
    return signals

