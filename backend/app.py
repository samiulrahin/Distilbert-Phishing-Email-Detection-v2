from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.config import Settings, settings
from backend.model_loader import ModelService
from backend.privacy import PrivacySafeMetricsLogger
from backend.schemas import HealthResponse, PredictionRequest, PredictionResponse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = PROJECT_ROOT / "demo"


def create_app(app_settings: Settings = settings) -> FastAPI:
    model_service = ModelService(
        model_dir=app_settings.model_dir,
        baseline_path=app_settings.baseline_path,
        threshold=app_settings.decision_threshold,
        allow_demo_fallback=app_settings.allow_demo_fallback,
    )
    metrics_logger = PrivacySafeMetricsLogger(
        app_settings.log_path, enabled=app_settings.enable_metrics_log
    )

    app = FastAPI(
        title="Local Phishing Email Detector",
        version="0.1.0",
        description="Local-only research prototype for controlled email classification.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^(chrome-extension://.*|http://127\.0\.0\.1(?::\d+)?|http://localhost(?::\d+)?)$",
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.mount("/demo/static", StaticFiles(directory=DEMO_DIR), name="demo-static")

    @app.get("/demo", include_in_schema=False)
    def demo() -> FileResponse:
        return FileResponse(DEMO_DIR / "index.html")

    @app.get("/demo/icon.png", include_in_schema=False)
    def demo_icon() -> FileResponse:
        return FileResponse(PROJECT_ROOT / "extension" / "icons" / "icon48.png")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            model_name=model_service.model_name,
            model_version=model_service.model_version,
            model_mode=model_service.mode,
            model_ready=model_service.model_ready,
            policy_mode=model_service.policy_mode,
        )

    @app.post("/predict", response_model=PredictionResponse)
    def predict(payload: PredictionRequest) -> PredictionResponse:
        prediction = model_service.predict(payload.subject, payload.body)
        metrics_logger.record(
            source=payload.source,
            subject_chars=len(payload.subject),
            body_chars=len(payload.body),
            model_mode=model_service.mode,
            label=prediction.label,
            confidence=prediction.confidence,
            inference_ms=prediction.inference_ms,
        )
        return PredictionResponse(
            label=prediction.label,
            confidence=prediction.confidence,
            phishing_probability=prediction.phishing_probability,
            raw_model_score=prediction.raw_model_score,
            threshold=model_service.phishing_min,
            legitimate_max=model_service.legitimate_max,
            phishing_min=model_service.phishing_min,
            calibration_temperature=model_service.temperature,
            policy_mode=model_service.policy_mode,
            inference_ms=prediction.inference_ms,
            model_name=model_service.model_name,
            model_version=model_service.model_version,
            model_mode=model_service.mode,
            risk_signals=prediction.risk_signals,
            limitation=model_service.limitation,
        )

    app.state.model_service = model_service
    return app


app = create_app()
