from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class PredictionRequest(BaseModel):
    subject: str = Field(default="", max_length=1000)
    body: str = Field(default="", max_length=50000)
    source: Literal["controlled-gmail-test", "paste-mode", "api-test"] = (
        "controlled-gmail-test"
    )

    @model_validator(mode="after")
    def require_text(self) -> "PredictionRequest":
        if not self.subject.strip() and not self.body.strip():
            raise ValueError("A subject or email body is required.")
        return self


class RiskSignal(BaseModel):
    category: str
    description: str


class PredictionResponse(BaseModel):
    label: Literal["phishing", "legitimate", "uncertain"]
    confidence: float = Field(ge=0.0, le=1.0)
    phishing_probability: float = Field(ge=0.0, le=1.0)
    raw_model_score: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    legitimate_max: float = Field(ge=0.0, le=1.0)
    phishing_min: float = Field(ge=0.0, le=1.0)
    calibration_temperature: float = Field(gt=0.0)
    policy_mode: str
    inference_ms: float = Field(ge=0.0)
    model_name: str
    model_version: str
    model_mode: Literal["distilbert", "baseline", "demonstration"]
    privacy: Literal["local-only"] = "local-only"
    risk_signals: list[RiskSignal] = Field(default_factory=list)
    limitation: str


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    model_name: str
    model_version: str
    model_mode: Literal["distilbert", "baseline", "demonstration"]
    model_ready: bool
    policy_mode: str
    privacy: Literal["local-only"] = "local-only"
