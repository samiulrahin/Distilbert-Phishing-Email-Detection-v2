from __future__ import annotations

import pandas as pd

from ml.preprocess import prepare_dataframe, stratified_splits


def test_preparation_removes_duplicates_invalid_labels_and_conflicts():
    frame = pd.DataFrame(
        {
            "Email Text": [
                "Subject: Team meeting\nThe agenda is attached.",
                "Subject: Team meeting\nThe agenda is attached.",
                "Subject: Reset now\nClick https://example.invalid and enter password.",
                "Subject: Reset now\nClick https://example.invalid and enter password.",
                "Too short",
                "A valid body but unsupported label",
            ],
            "Email Type": ["Safe Email", "Safe Email", "Phishing Email", "Safe Email", "Safe Email", "unknown"],
        }
    )
    prepared, report = prepare_dataframe(frame)
    assert len(prepared) == 1
    assert prepared.iloc[0]["label"] == 0
    assert report["exact_duplicates_removed"] == 1
    assert report["conflicting_label_rows_removed"] == 2
    assert report["invalid_label_rows_removed"] == 1


def test_stratified_split_is_reproducible_and_disjoint():
    frame = pd.DataFrame(
        {
            "example_id": [f"id-{index}" for index in range(100)],
            "subject": [""] * 100,
            "body": [f"message {index}" for index in range(100)],
            "text": [f"message {index}" for index in range(100)],
            "label": [index % 2 for index in range(100)],
        }
    )
    first = stratified_splits(frame, seed=42)
    second = stratified_splits(frame, seed=42)
    assert [set(split.example_id) for split in first] == [set(split.example_id) for split in second]
    assert set(first[0].example_id).isdisjoint(first[1].example_id)
    assert set(first[0].example_id).isdisjoint(first[2].example_id)
    assert [len(split) for split in first] == [70, 15, 15]

