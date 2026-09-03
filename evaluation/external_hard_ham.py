from __future__ import annotations

import argparse
import email
import hashlib
import json
import tarfile
from email import policy
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from backend.model_loader import ModelService
from ml.policy import calibrate_probability, triage_label
from ml.preprocess import clean_text, stable_hash


PROJECT_ROOT = Path(__file__).resolve().parents[1]
URL = "https://spamassassin.apache.org/old/publiccorpus/20030228_hard_ham.tar.bz2"
EXPECTED_SHA256 = "ce2ce67880643dbde65ea7f85bffbfe4417349c4bd80b6b0de56262ae6b0a9c9"


def download(path: Path) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(URL, timeout=60)
        response.raise_for_status()
        path.write_bytes(response.content)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(f"Unexpected corpus SHA-256: {digest}")


def extract_messages(path: Path) -> list[dict[str, str]]:
    rows = []
    with tarfile.open(path, "r:bz2") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            raw = archive.extractfile(member).read()
            message = email.message_from_bytes(raw, policy=policy.default)
            subject = clean_text(str(message.get("subject") or ""))
            parts = []
            if message.is_multipart():
                for part in message.walk():
                    if part.get_content_type() == "text/plain" and part.get_content_disposition() != "attachment":
                        try:
                            parts.append(part.get_content())
                        except (KeyError, LookupError, UnicodeError):
                            continue
            else:
                try:
                    parts.append(message.get_content())
                except (KeyError, LookupError, UnicodeError):
                    parts.append(raw.decode("utf-8", "replace"))
            body = clean_text("\n".join(parts))
            if len(body) >= 10:
                rows.append({"source_id": member.name, "subject": subject, "body": body, "body_hash": stable_hash(body)})
    return rows


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    download(args.archive)
    existing_ids = set()
    for split in ("train", "validation", "test"):
        frame = pd.read_csv(PROJECT_ROOT / "data" / "splits" / f"{split}.csv", usecols=["example_id"])
        existing_ids.update(frame["example_id"].astype(str))
    rows = [row for row in extract_messages(args.archive) if row["body_hash"][:16] not in existing_ids]

    policy_data = json.loads((args.model_dir / "decision_policy.json").read_text(encoding="utf-8"))
    service = ModelService(args.model_dir, PROJECT_ROOT / "models" / "missing-baseline.joblib", allow_demo_fallback=False)
    results = []
    for row in rows:
        raw = service.predict_raw(row["subject"], row["body"])
        calibrated = float(calibrate_probability(raw, policy_data["temperature"]))
        label = triage_label(calibrated, policy_data["legitimate_max"], policy_data["phishing_min"])
        results.append(
            {
                "source_id_hash": hashlib.sha256(row["source_id"].encode("utf-8")).hexdigest()[:16],
                "expected_label": "legitimate",
                "raw_model_score": round(raw, 6),
                "calibrated_risk_score": round(calibrated, 6),
                "predicted_label": label,
            }
        )
    frame = pd.DataFrame(results)
    metrics = {
        "source": URL,
        "archive_sha256": EXPECTED_SHA256,
        "corpus_description": "Apache SpamAssassin hard_ham; historical legitimate messages designed to resemble spam.",
        "parsed_rows_after_exact_overlap_removal": len(frame),
        "exact_overlap_rows_removed": len(extract_messages(args.archive)) - len(frame),
        "legitimate_rate": float(np.mean(frame.predicted_label == "legitimate")),
        "uncertain_rate": float(np.mean(frame.predicted_label == "uncertain")),
        "false_positive_rate": float(np.mean(frame.predicted_label == "phishing")),
        "copyright_control": "Raw messages remain local and are not redistributed; output contains hashed source IDs only.",
        "validity_limit": "The corpus is historical and measures cross-source hard-ham behaviour, not modern Gmail prevalence.",
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.results_dir / "v2_external_hard_ham_predictions.csv", index=False)
    (args.results_dir / "v2_external_hard_ham_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Version 2 on the official Apache hard-ham corpus.")
    parser.add_argument("--archive", type=Path, default=PROJECT_ROOT / "tmp" / "external" / "spamassassin" / "20030228_hard_ham.tar.bz2")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models" / "distilbert-phishing-v2")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "evaluation" / "results")
    return parser


if __name__ == "__main__":
    print(json.dumps(evaluate(build_parser().parse_args()), indent=2))
