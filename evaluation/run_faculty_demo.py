from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = PROJECT_ROOT / "evaluation" / "faculty_demo_emails.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation" / "results" / "v2_faculty_demo_results.json"


def run_demo(base_url: str, cases_path: Path, output_path: Path) -> dict:
    health_response = requests.get(f"{base_url}/health", timeout=15)
    health_response.raise_for_status()
    health = health_response.json()

    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    results = []
    for case in cases:
        response = requests.post(
            f"{base_url}/predict",
            json={
                "subject": case["subject"],
                "body": case["body"],
                "source": "api-test",
            },
            timeout=30,
        )
        response.raise_for_status()
        prediction = response.json()
        results.append(
            {
                "id": case["id"],
                "expected_label": case["expected_label"],
                "actual_label": prediction["label"],
                "matched": prediction["label"] == case["expected_label"],
                "calibrated_risk_score": prediction["phishing_probability"],
                "raw_model_score": prediction["raw_model_score"],
                "inference_ms": prediction["inference_ms"],
            }
        )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Controlled synthetic faculty demonstration; no personal email data.",
        "service": health,
        "all_expected_states_matched": all(item["matched"] for item in results),
        "cases": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the controlled V2 faculty demonstration.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = run_demo(args.base_url.rstrip("/"), args.cases, args.output)
    print(json.dumps(report, indent=2))
    if not report["all_expected_states_matched"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
