"""
Tests for dataset loader eligibility gate (SLAI-001 / SLAI-002 / SLAI-017).

All tests use temporary JSONL files via pytest's tmp_path fixture and
monkeypatch settings.data_dir / settings.krio_data_dir so no real dataset
files are required.
"""
from __future__ import annotations

import json
from pathlib import Path

from sierraleone_ai.datasets.loader import LoadPolicy, _apply_policy, _load_jsonl, dataset_summary

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _record(
    source: str = "test_source",
    status: str = "unvalidated",
    consent: bool = True,
    split: str = "train",
) -> dict:
    return {
        "metadata": {
            "record_id": "00000000-0000-0000-0000-000000000000",
            "language": "krio",
            "source": source,
            "collection_date": "2024-01-01",
            "licence": "CC-BY-4.0",
            "validation_status": status,
            "dataset_version": "0.1.0",
            "split": split,
            "contributor_consent": consent,
            "consent_scope": ["training"] if consent else [],
        },
        "text": "TEST_FIXTURE_TEXT_LOADER",
        "script": "Latin",
    }


# ---------------------------------------------------------------------------
# _apply_policy tests
# ---------------------------------------------------------------------------

class TestApplyPolicy:
    def test_rejected_records_always_excluded(self):
        raw = [_record(status="rejected")]
        admitted, skipped = _apply_policy(raw, "train", LoadPolicy(), "test.jsonl")
        assert len(admitted) == 0
        assert skipped["rejected"] == 1

    def test_partially_validated_excluded_when_validated_only(self):
        raw = [_record(status="partially_validated")]
        admitted, _ = _apply_policy(raw, "train", LoadPolicy(validated_only=True), "test.jsonl")
        assert len(admitted) == 0

    def test_partially_validated_allowed_when_not_validated_only(self):
        raw = [_record(status="partially_validated", consent=True)]
        admitted, _ = _apply_policy(raw, "train", LoadPolicy(validated_only=False), "test.jsonl")
        assert len(admitted) == 1

    def test_no_consent_excluded_when_require_consent(self):
        raw = [_record(consent=False)]
        admitted, skipped = _apply_policy(raw, "train", LoadPolicy(require_consent=True), "test.jsonl")
        assert len(admitted) == 0
        assert skipped["no_consent"] == 1

    def test_no_consent_allowed_when_consent_not_required(self):
        raw = [_record(consent=False, status="validated")]
        admitted, _ = _apply_policy(raw, "train", LoadPolicy(require_consent=False), "test.jsonl")
        assert len(admitted) == 1

    def test_template_records_excluded_by_default(self):
        raw = [_record(source="SCHEMA_TEMPLATE_NOT_TRAINING_DATA")]
        admitted, skipped = _apply_policy(raw, "train", LoadPolicy(), "test.jsonl")
        assert len(admitted) == 0
        assert skipped["template_record"] == 1

    def test_template_allowed_when_policy_permits(self):
        raw = [_record(source="SCHEMA_TEMPLATE_NOT_TRAINING_DATA", consent=False)]
        admitted, _ = _apply_policy(raw, "train", LoadPolicy(allow_templates=True, require_consent=False), "test.jsonl")
        assert len(admitted) == 1

    def test_split_mismatch_excluded(self):
        raw = [_record(split="test")]  # loading "train" split
        admitted, skipped = _apply_policy(raw, "train", LoadPolicy(require_consent=False), "test.jsonl")
        assert len(admitted) == 0
        assert any("split_mismatch" in k for k in skipped)

    def test_machine_generated_excluded_by_default(self):
        raw = [_record(status="machine_generated", consent=False)]
        admitted, skipped = _apply_policy(raw, "train", LoadPolicy(require_consent=False), "test.jsonl")
        assert len(admitted) == 0
        assert skipped["machine_generated"] == 1

    def test_machine_generated_allowed_when_policy_permits(self):
        raw = [_record(status="machine_generated", consent=False)]
        admitted, _ = _apply_policy(
            raw, "train",
            LoadPolicy(include_machine_generated=True, require_consent=False),
            "test.jsonl",
        )
        assert len(admitted) == 1


class TestLoadJsonl:
    def test_load_valid_jsonl(self, tmp_path: Path):
        path = tmp_path / "data.jsonl"
        _write_jsonl(path, [{"a": 1}, {"b": 2}])
        records = _load_jsonl(path)
        assert len(records) == 2

    def test_bad_json_line_skipped(self, tmp_path: Path):
        path = tmp_path / "data.jsonl"
        path.write_text('{"valid": true}\nNOT_JSON\n{"also": true}\n', encoding="utf-8")
        records = _load_jsonl(path)
        assert len(records) == 2

    def test_blank_lines_and_comments_skipped(self, tmp_path: Path):
        path = tmp_path / "data.jsonl"
        path.write_text('\n# comment\n{"x": 1}\n\n{"y": 2}\n', encoding="utf-8")
        records = _load_jsonl(path)
        assert len(records) == 2


class TestDatasetSummary:
    def test_template_files_not_counted(self, tmp_path: Path, monkeypatch):
        """example_* files and SCHEMA_TEMPLATE records must not appear in summary counts."""
        import sierraleone_ai.config as cfg_module

        krio_cleaned = tmp_path / "krio" / "cleaned"
        krio_cleaned.mkdir(parents=True)

        # Write an example_ file (should be skipped by name)
        _write_jsonl(
            krio_cleaned / "example_krio_text.jsonl",
            [_record(source="SCHEMA_TEMPLATE_NOT_TRAINING_DATA", consent=False)],
        )
        # Write a template-source record in a regular file (should be skipped by content)
        _write_jsonl(
            krio_cleaned / "krio_text_train.jsonl",
            [_record(source="SCHEMA_TEMPLATE_NOT_TRAINING_DATA", consent=False)],
        )

        monkeypatch.setattr(cfg_module.settings, "data_dir", tmp_path)
        summary = dataset_summary()
        # krio/cleaned should not appear (all records filtered out)
        assert summary.get("krio", {}).get("cleaned", 0) == 0

    def test_real_records_counted(self, tmp_path: Path, monkeypatch):
        import sierraleone_ai.config as cfg_module

        krio_cleaned = tmp_path / "krio" / "cleaned"
        krio_cleaned.mkdir(parents=True)

        _write_jsonl(
            krio_cleaned / "krio_text_train.jsonl",
            [_record(source="real_collection_2024", consent=False)] * 5,
        )

        monkeypatch.setattr(cfg_module.settings, "data_dir", tmp_path)
        summary = dataset_summary()
        assert summary.get("krio", {}).get("cleaned", 0) == 5

