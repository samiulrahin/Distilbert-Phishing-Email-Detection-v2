from __future__ import annotations

import math
import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from backend.risk_signals import identify_risk_signals
from ml.policy import calibrate_probability, triage_label


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    phishing_probability: float
    raw_model_score: float
    inference_ms: float
    risk_signals: list[dict[str, str]]


class ModelService:
    """Loads DistilBERT once, with explicit baseline and demo fallbacks."""

    def __init__(
        self,
        model_dir: Path,
        baseline_path: Path,
        threshold: float = 0.5,
        allow_demo_fallback: bool = True,
    ) -> None:
        self.model_dir = model_dir
        self.baseline_path = baseline_path
        self.threshold = threshold
        self.allow_demo_fallback = allow_demo_fallback
        self.mode = "demonstration"
        self.model_name = "controlled-rule-demonstrator"
        self.model_version = "demo-v1"
        self.model_ready = False
        self.policy_mode = "legacy-threshold"
        self.temperature = 1.0
        self.legitimate_max = max(0.0, threshold - 0.05)
        self.phishing_min = min(1.0, threshold + 0.05)
        self.limitation = (
            "Demonstration rules are active. This output is not a trained model result."
        )
        self._model = None
        self._tokenizer = None
        self._device = None
        self._lock = threading.Lock()
        self._load_best_available_model()

    def _load_best_available_model(self) -> None:
        if self._has_transformer_checkpoint(self.model_dir):
            try:
                self._load_distilbert()
                return
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                self.limitation = f"DistilBERT checkpoint could not load: {type(exc).__name__}."

        if self.baseline_path.exists():
            try:
                self._load_baseline()
                return
            except (ImportError, OSError, ValueError) as exc:
                self.limitation = f"Baseline model could not load: {type(exc).__name__}."

        if not self.allow_demo_fallback:
            raise RuntimeError("No trained model is available and demo fallback is disabled.")

    @staticmethod
    def _has_transformer_checkpoint(path: Path) -> bool:
        model_file = (path / "model.safetensors").exists() or (
            path / "pytorch_model.bin"
        ).exists()
        return model_file and (path / "config.json").exists()

    def _load_distilbert(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self._model.to(self._device)
        self._model.eval()
        threshold_path = self.model_dir / "decision_threshold.json"
        if threshold_path.exists():
            self.threshold = float(json.loads(threshold_path.read_text(encoding="utf-8"))["threshold"])
        metadata_path = self.model_dir / "research_metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.model_name = str(metadata.get("model_name", self.model_name))
            self.model_version = str(metadata.get("model_version", self.model_version))
        policy_path = self.model_dir / "decision_policy.json"
        if policy_path.exists():
            policy = json.loads(policy_path.read_text(encoding="utf-8"))
            self.temperature = float(policy["temperature"])
            self.legitimate_max = float(policy["legitimate_max"])
            self.phishing_min = float(policy["phishing_min"])
            self.policy_mode = str(policy.get("version", "selective"))
        else:
            self._configure_legacy_policy()
        self.mode = "distilbert"
        if self.model_name == "controlled-rule-demonstrator":
            self.model_name = "distilbert-phishing-email"
        if self.model_version == "demo-v1":
            self.model_version = "v1"
        self.model_ready = True
        self.limitation = (
            "Content-only research prototype. The score does not authenticate the sender, links, or message headers; review uncertain results manually."
        )

    def _load_baseline(self) -> None:
        import joblib

        self._model = joblib.load(self.baseline_path)
        threshold_path = self.baseline_path.parent / "baseline_threshold.json"
        if threshold_path.exists():
            self.threshold = float(json.loads(threshold_path.read_text(encoding="utf-8"))["threshold"])
        self._configure_legacy_policy()
        self.mode = "baseline"
        self.model_name = "tfidf-logistic-regression"
        self.model_version = "v1"
        self.model_ready = True
        self.limitation = (
            "Classical baseline is active; DistilBERT checkpoint is not currently loaded."
        )

    def predict(self, subject: str, body: str) -> Prediction:
        text = self._combine(subject, body)
        start = perf_counter()
        with self._lock:
            raw_probability = self._predict_raw(text)
        elapsed_ms = (perf_counter() - start) * 1000
        probability = float(calibrate_probability(raw_probability, self.temperature))
        label = triage_label(probability, self.legitimate_max, self.phishing_min)
        confidence = 0.5 if label == "uncertain" else (
            probability if label == "phishing" else 1.0 - probability
        )
        return Prediction(
            label=label,
            confidence=round(float(confidence), 6),
            phishing_probability=round(float(probability), 6),
            raw_model_score=round(float(raw_probability), 6),
            inference_ms=round(elapsed_ms, 3),
            risk_signals=identify_risk_signals(text),
        )

    def predict_raw(self, subject: str, body: str) -> float:
        text = self._combine(subject, body)
        with self._lock:
            return float(self._predict_raw(text))

    def _predict_raw(self, text: str) -> float:
        if self.mode == "distilbert":
            return self._predict_distilbert(text)
        if self.mode == "baseline":
            return self._predict_baseline(text)
        return self._predict_demonstration(text)

    def _configure_legacy_policy(self) -> None:
        margin = min(0.08, self.threshold * 0.5, (1.0 - self.threshold) * 0.5)
        self.legitimate_max = max(0.0, self.threshold - margin)
        self.phishing_min = min(1.0, self.threshold + margin)
        self.temperature = 1.0
        self.policy_mode = "legacy-threshold"

    @staticmethod
    def _combine(subject: str, body: str) -> str:
        subject = re.sub(r"\s+", " ", subject).strip()
        body = re.sub(r"\s+", " ", body).strip()
        return f"Subject: {subject}\n\nBody: {body}".strip()

    def _predict_distilbert(self, text: str) -> float:
        import torch

        encoded = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=False,
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = self._model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1)[0]
        id2label = {int(k): str(v).lower() for k, v in self._model.config.id2label.items()}
        phishing_index = next(
            (index for index, label in id2label.items() if "phish" in label), 1
        )
        return probabilities[phishing_index].item()

    def _predict_baseline(self, text: str) -> float:
        classes = list(self._model.classes_)
        probabilities = self._model.predict_proba([text])[0]
        phishing_index = classes.index(1) if 1 in classes else classes.index("phishing")
        return float(probabilities[phishing_index])

    @staticmethod
    def _predict_demonstration(text: str) -> float:
        signals = identify_risk_signals(text, limit=10)
        score = -1.9 + (0.82 * len(signals))
        if re.search(r"\b(meeting|minutes|agenda|project update|thank you|attached report)\b", text, re.I):
            score -= 0.65
        return 1.0 / (1.0 + math.exp(-score))
