from __future__ import annotations

import argparse
import hashlib
import json
import re
from html import unescape
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "Phishing_Email.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "splits"

TEXT_COLUMNS = ("email text", "text", "email", "body", "message")
LABEL_COLUMNS = ("email type", "label", "class", "type", "category")
SUBJECT_PATTERN = re.compile(r"(?:^|\n)\s*(?:email )?subject\s*:\s*(.+?)(?:\n|$)", re.I)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"[ \t\f\v]+")
MULTI_NEWLINE_PATTERN = re.compile(r"\n{3,}")


def canonical_column(columns: list[str], candidates: tuple[str, ...]) -> str:
    lookup = {column.strip().lower(): column for column in columns}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    raise ValueError(f"Could not find any of these columns: {', '.join(candidates)}")


def normalise_label(value: object) -> int | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"1", "phishing", "phishing email", "spam", "malicious", "malware"}:
        return 1
    if text in {"0", "safe", "safe email", "legitimate", "legitimate email", "benign", "ham"}:
        return 0
    return None


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = unescape(str(value).replace("\x00", " ").replace("\r\n", "\n"))
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text)
    text = MULTI_NEWLINE_PATTERN.sub("\n\n", text)
    return text.strip()


def extract_subject(text: str) -> str:
    match = SUBJECT_PATTERN.search(text)
    return clean_text(match.group(1))[:500] if match else ""


def stable_hash(text: str) -> str:
    normalised = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def prepare_dataframe(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    text_column = canonical_column(list(frame.columns), TEXT_COLUMNS)
    label_column = canonical_column(list(frame.columns), LABEL_COLUMNS)
    prepared = pd.DataFrame(
        {
            "body": frame[text_column].map(clean_text),
            "label": frame[label_column].map(normalise_label),
        }
    )
    initial_rows = len(prepared)
    invalid_label_rows = int(prepared["label"].isna().sum())
    prepared = prepared.dropna(subset=["label"]).copy()
    prepared["label"] = prepared["label"].astype(int)
    empty_rows = int((prepared["body"].str.len() < 10).sum())
    prepared = prepared[prepared["body"].str.len() >= 10].copy()
    prepared["text_hash"] = prepared["body"].map(stable_hash)

    conflicts = prepared.groupby("text_hash")["label"].nunique()
    conflicting_hashes = set(conflicts[conflicts > 1].index)
    conflicting_rows = int(prepared["text_hash"].isin(conflicting_hashes).sum())
    prepared = prepared[~prepared["text_hash"].isin(conflicting_hashes)].copy()
    before_dedup = len(prepared)
    prepared = prepared.drop_duplicates(subset=["text_hash"], keep="first").copy()
    duplicate_rows = before_dedup - len(prepared)

    prepared["subject"] = prepared["body"].map(extract_subject)
    prepared["text"] = (
        "Subject: " + prepared["subject"] + "\n\nBody: " + prepared["body"]
    )
    prepared["example_id"] = prepared["text_hash"].str[:16]
    prepared = prepared[["example_id", "subject", "body", "text", "label"]]

    report = {
        "initial_rows": initial_rows,
        "invalid_label_rows_removed": invalid_label_rows,
        "empty_or_short_rows_removed": empty_rows,
        "conflicting_label_rows_removed": conflicting_rows,
        "exact_duplicates_removed": duplicate_rows,
        "final_rows": len(prepared),
        "class_distribution": {
            str(key): int(value)
            for key, value in prepared["label"].value_counts().sort_index().items()
        },
        "source_text_column": text_column,
        "source_label_column": label_column,
    }
    return prepared.reset_index(drop=True), report


def stratified_splits(
    frame: pd.DataFrame, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, remainder = train_test_split(
        frame, test_size=0.30, random_state=seed, stratify=frame["label"]
    )
    validation, test = train_test_split(
        remainder,
        test_size=0.50,
        random_state=seed,
        stratify=remainder["label"],
    )
    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)


def run(input_path: Path, output_dir: Path, seed: int = 42) -> dict[str, object]:
    frame = pd.read_csv(input_path, encoding_errors="replace")
    prepared, report = prepare_dataframe(frame)
    train, validation, test = stratified_splits(prepared, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", train), ("validation", validation), ("test", test)):
        split.to_csv(output_dir / f"{name}.csv", index=False)
    report["split_seed"] = seed
    report["split_rows"] = {
        "train": len(train),
        "validation": len(validation),
        "test": len(test),
    }
    (output_dir / "data_quality_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean, deduplicate, and split email data.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output_dir, args.seed), indent=2))


if __name__ == "__main__":
    main()

