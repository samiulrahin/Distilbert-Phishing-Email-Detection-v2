from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = (
    "https://huggingface.co/datasets/zefang-liu/phishing-email-dataset/"
    "resolve/main/Phishing_Email.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "Phishing_Email.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, output: Path, force: bool = False) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        return {
            "status": "existing",
            "path": str(output),
            "bytes": output.stat().st_size,
            "sha256": sha256(output),
        }

    partial = output.with_suffix(output.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "UWE-CSCT-research-prototype/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)
    partial.replace(output)
    return {
        "status": "downloaded",
        "path": str(output),
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the documented public email dataset.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = download(args.url, args.output, args.force)
    result.update({"source_url": args.url, "accessed_utc": datetime.now(UTC).isoformat()})
    metadata_path = args.output.parent / "download_metadata.json"
    metadata_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

