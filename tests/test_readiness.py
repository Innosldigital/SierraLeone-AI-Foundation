"""
Tests for training readiness check (SLAI-002 / SLAI-017).

Verifies that:
- With only template files present, all readiness items are NOT ready.
- require_training_approval raises even when allow_training=True if datasets are empty.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sierraleone_ai.datasets.manifest import (
    build_manifest,
    scan_eligible_records,
)
from sierraleone_ai.training.guard import (
    ReadinessItem,
    check_dataset_readiness,
    require_training_approval,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _template_record() -> dict:
    """A SCHEMA_TEMPLATE record — must never count towards readiness."""
    return {
        "metadata": {
            "record_id": "00000000-0000-0000-0000-000000000000",
            "language": "krio",
            "source": "SCHEMA_TEMPLATE_NOT_TRAINING_DATA",
            "collection_date": "2024-01-01",
            "licence": "CC-BY-4.0",
            "validation_status": "unvalidated",
            "dataset_version": "0.1.0",
            "split": "train",
            "contributor_consent": False,
        },
        "text": "SCHEMA_TEMPLATE_NOT_TRAINING_DATA",
    }


def _valid_record(
    record_id: str = "11111111-1111-4111-8111-111111111111",
    source: str = "test_fixture_approved_source",
    split: str = "train",
) -> dict:
    return {
        "metadata": {
            "record_id": record_id,
            "language": "krio",
            "source": source,
            "collection_date": "2024-01-01",
            "licence": "CC-BY-4.0",
            "licence_id": "CC-BY-4.0",
            "validation_status": "validated",
            "dataset_version": "0.1.0",
            "split": split,
            "contributor_consent": True,
            "consent_scope": ["text_training"],
            "access_class": "public",
            "redistribution_permitted": True,
            "training_permitted": True,
            "reviews": [
                {
                    "reviewer_id": "TEST_REV_01",
                    "is_native_speaker_of_record_language": True,
                    "reviewed_on": "2024-01-02",
                    "verdict": "approve",
                },
                {
                    "reviewer_id": "TEST_REV_02",
                    "is_native_speaker_of_record_language": True,
                    "reviewed_on": "2024-01-02",
                    "verdict": "approve",
                },
            ],
        },
        "text": "TEST_FIXTURE_VALID_RECORD",
        "script": "Latin",
    }


class TestCheckDatasetReadiness:
    def test_empty_data_dir_all_not_ready(self, tmp_path: Path, monkeypatch):
        """With no dataset files at all, every readiness item must be NOT ready."""
        import sierraleone_ai.config as cfg_module

        monkeypatch.setattr(cfg_module.settings, "data_dir", tmp_path)

        items = check_dataset_readiness()
        # Returns a dict[str, ReadinessItem]
        assert isinstance(items, dict)
        assert all(isinstance(item, ReadinessItem) for item in items.values())
        not_ready = [item for item in items.values() if item.ok]
        assert not not_ready, (
            f"Expected all items NOT ready, but these passed: {[i.key for i in not_ready]}"
        )

    def test_template_only_all_not_ready(self, tmp_path: Path, monkeypatch):
        """Template-only JSONL files must NOT satisfy readiness thresholds."""
        import sierraleone_ai.config as cfg_module

        krio_dir = tmp_path / "krio"
        monkeypatch.setattr(cfg_module.settings, "data_dir", tmp_path)

        # Place template records in all expected locations
        for subdir, fname in [
            ("validated", "krio_text_train.jsonl"),
            ("conversations", "conversations_train.jsonl"),
            ("translations", "translations_english_krio_train.jsonl"),
        ]:
            _write_jsonl(krio_dir / subdir / fname, [_template_record()] * 10)

        items = check_dataset_readiness()
        ready = [item for item in items.values() if item.ok]
        assert not ready, (
            f"Template records must not count. Ready items: {[(i.key, i.have) for i in ready]}"
        )

    def test_readiness_item_properties(self, tmp_path: Path, monkeypatch):
        """ReadinessItem must have canonical key, ok, have, need properties (PA-021)."""
        import sierraleone_ai.config as cfg_module

        monkeypatch.setattr(cfg_module.settings, "data_dir", tmp_path)
        items = check_dataset_readiness()
        for item in items.values():
            assert isinstance(item.key, str)
            assert isinstance(item.have, int)
            assert isinstance(item.need, int)
            assert item.ok == (item.have >= item.need)

    def test_ineligible_records_count_zero(self, tmp_path: Path, monkeypatch):
        """PA-001: 3 unvalidated/no-consent/unknown-licence/wrong-split + 1 machine_generated -> count 0."""
        import sierraleone_ai.config as cfg_module

        krio_dir = tmp_path / "krio"
        monkeypatch.setattr(cfg_module.settings, "data_dir", tmp_path)

        ineligible_rows = [
            # 3 unvalidated / no consent / unknown licence / wrong split
            {
                "metadata": {
                    "record_id": f"00000000-0000-4000-8000-{i:012d}",
                    "language": "krio",
                    "source": "test_fixture_approved_source",
                    "collection_date": "2024-01-01",
                    "licence": "unknown",
                    "licence_id": "unknown",
                    "validation_status": "unvalidated",
                    "dataset_version": "0.1.0",
                    "split": "test",
                    "contributor_consent": False,
                    "training_permitted": False,
                },
                "text": f"TEST_FIXTURE_INELIGIBLE_{i}",
                "script": "Latin",
            }
            for i in range(1, 4)
        ] + [
            # 1 machine generated
            {
                "metadata": {
                    "record_id": "00000000-0000-4000-8000-000000000099",
                    "language": "krio",
                    "source": "test_fixture_approved_source",
                    "collection_date": "2024-01-01",
                    "licence": "CC-BY-4.0",
                    "licence_id": "CC-BY-4.0",
                    "validation_status": "machine_generated",
                    "dataset_version": "0.1.0",
                    "split": "train",
                    "contributor_consent": True,
                    "consent_scope": ["text_training"],
                    "training_permitted": True,
                },
                "text": "TEST_FIXTURE_MACHINE_GEN",
                "script": "Latin",
            }
        ]

        _write_jsonl(krio_dir / "cleaned" / "krio_text_train.jsonl", ineligible_rows)
        scan = scan_eligible_records("krio", "train")
        assert scan.eligible_counts["text_training"] == 0

        manifest_path = build_manifest("krio", "train")
        items = check_dataset_readiness("krio", "train", manifest_path=manifest_path)
        assert items["cleaned_text"].have == 0

    def test_quarantine_issue_file_counts_zero(self, tmp_path: Path, monkeypatch):
        """PA-007: *.issues.jsonl quarantine reports are not counted as data."""
        import sierraleone_ai.config as cfg_module

        krio_dir = tmp_path / "krio"
        monkeypatch.setattr(cfg_module.settings, "data_dir", tmp_path)

        issue_rows = [
            {"path": "foo.jsonl", "line": 1, "record_id": None, "error": "malformed"},
            {"path": "foo.jsonl", "line": 2, "record_id": None, "error": "malformed"},
        ]
        _write_jsonl(krio_dir / "quarantine" / "sample.issues.jsonl", issue_rows)
        _write_jsonl(krio_dir / "cleaned" / "krio_text_train.issues.jsonl", issue_rows)

        scan = scan_eligible_records("krio", "train")
        assert scan.eligible_counts["text_training"] == 0
        assert len(scan.files) == 0

    def test_same_record_id_in_cleaned_and_validated_counts_once(self, tmp_path: Path, monkeypatch):
        """PA-001: Same record_id present in both cleaned/ and validated/ must not double count."""
        import sierraleone_ai.config as cfg_module

        krio_dir = tmp_path / "krio"
        monkeypatch.setattr(cfg_module.settings, "data_dir", tmp_path)

        row = _valid_record(record_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
        _write_jsonl(krio_dir / "cleaned" / "krio_text_train.jsonl", [row])
        _write_jsonl(krio_dir / "validated" / "krio_text_train.jsonl", [row])

        scan = scan_eligible_records("krio", "train")
        assert scan.eligible_counts["text_training"] == 1

        manifest_path = build_manifest("krio", "train")
        items = check_dataset_readiness("krio", "train", manifest_path=manifest_path)
        assert items["cleaned_text"].have == 1

    def test_manifest_hash_mismatch_invalidates_readiness(self, tmp_path: Path, monkeypatch):
        """PA-001: Tampering with a data file after manifest build invalidates readiness counts."""
        import sierraleone_ai.config as cfg_module

        krio_dir = tmp_path / "krio"
        monkeypatch.setattr(cfg_module.settings, "data_dir", tmp_path)

        row = _valid_record()
        target = krio_dir / "validated" / "krio_text_train.jsonl"
        _write_jsonl(target, [row])

        manifest_path = build_manifest("krio", "train")
        items = check_dataset_readiness("krio", "train", manifest_path=manifest_path)
        assert items["cleaned_text"].have == 1

        # Tamper with file
        target.write_text(target.read_text(encoding="utf-8") + "\n# extra\n", encoding="utf-8")
        stale_items = check_dataset_readiness("krio", "train", manifest_path=manifest_path)
        assert stale_items["cleaned_text"].have == 0
        assert "stale" in stale_items["cleaned_text"].detail

    def test_readiness_turns_green_when_threshold_met(self, tmp_path: Path, monkeypatch):
        """PA-001: Readiness item turns ok=True when threshold is satisfied by manifest."""
        import sierraleone_ai.config as cfg_module
        import sierraleone_ai.training.guard as guard_mod

        krio_dir = tmp_path / "krio"
        monkeypatch.setattr(cfg_module.settings, "data_dir", tmp_path)
        monkeypatch.setattr(
            guard_mod, "_MIN_RECORDS", {**guard_mod._MIN_RECORDS, "cleaned_text": 2}
        )

        rows = [
            _valid_record(record_id="11111111-1111-4111-8111-111111111111"),
            _valid_record(record_id="22222222-2222-4222-8222-222222222222"),
        ]
        _write_jsonl(krio_dir / "validated" / "krio_text_train.jsonl", rows)

        manifest_path = build_manifest("krio", "train")
        items = check_dataset_readiness("krio", "train", manifest_path=manifest_path)
        assert items["cleaned_text"].have == 2
        assert items["cleaned_text"].need == 2
        assert items["cleaned_text"].ok is True


class TestRequireTrainingApproval:
    def test_raises_by_default(self):
        with pytest.raises(RuntimeError, match="TRAINING GUARD"):
            require_training_approval("test task")

    def test_raises_even_with_allow_training_set_if_no_data(self, tmp_path: Path, monkeypatch):
        """
        Even with SLAI_ALLOW_TRAINING=true, the guard must still raise
        when dataset readiness thresholds are not met.
        """
        import sierraleone_ai.config as cfg_module

        monkeypatch.setattr(cfg_module.settings, "allow_training", True)
        monkeypatch.setattr(cfg_module.settings, "data_dir", tmp_path)

        with pytest.raises(RuntimeError, match="TRAINING GUARD"):
            require_training_approval("test task with empty data")
