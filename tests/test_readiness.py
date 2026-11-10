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
            f"Expected all items NOT ready, but these passed: "
            f"{[i.key for i in not_ready]}"
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
            f"Template records must not count. Ready items: "
            f"{[(i.key, i.have) for i in ready]}"
        )

    def test_readiness_item_has_name_alias(self, tmp_path: Path, monkeypatch):
        """ReadinessItem.name and .ready must be aliases for .key and .ok."""
        import sierraleone_ai.config as cfg_module
        monkeypatch.setattr(cfg_module.settings, "data_dir", tmp_path)
        items = check_dataset_readiness()
        for item in items.values():
            assert item.name == item.key
            assert item.ready == item.ok
            assert item.count == item.have


class TestRequireTrainingApproval:
    def test_raises_by_default(self):
        with pytest.raises(RuntimeError, match="TRAINING GUARD"):
            require_training_approval("test task")

    def test_raises_even_with_allow_training_set_if_no_data(
        self, tmp_path: Path, monkeypatch
    ):
        """
        Even with SLAI_ALLOW_TRAINING=true, the guard must still raise
        when dataset readiness thresholds are not met.
        """
        import sierraleone_ai.config as cfg_module
        monkeypatch.setattr(cfg_module.settings, "allow_training", True)
        monkeypatch.setattr(cfg_module.settings, "data_dir", tmp_path)

        with pytest.raises(RuntimeError, match="TRAINING GUARD"):
            require_training_approval("test task with empty data")
