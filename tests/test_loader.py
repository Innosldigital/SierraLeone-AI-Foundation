"""
Tests for dataset loader eligibility gate (SLAI-001 / SLAI-002 / SLAI-017).

All tests use temporary JSONL files via pytest's tmp_path fixture and
monkeypatch settings.data_dir / settings.krio_data_dir so no real dataset
files are required.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sierraleone_ai.datasets.loader import (
    DataValidationError,
    LoadPolicy,
    _apply_policy,
    _load_jsonl,
    dataset_summary,
    export_for_release,
    load_krio_text,
    scan_jsonl_for_pii,
)

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


class TestStrictAndPreviewLoad:
    """SLAI-022 / SLAI-025 / SLAI-031 tests."""

    def _valid_schema_dict(
        self,
        record_id: str = "11111111-1111-4111-8111-111111111111",
        source: str = "community_approved_2024",
        licence_id: str = "CC-BY-4.0",
        access_class: str = "public",
        redistribution_permitted: bool = True,
    ) -> dict:
        return {
            "metadata": {
                "record_id": record_id,
                "language": "krio",
                "source": source,
                "collection_date": "2024-01-01",
                "licence": "CC-BY-4.0",
                "licence_id": licence_id,
                "validation_status": "validated",
                "dataset_version": "0.1.0",
                "split": "train",
                "contributor_consent": True,
                "consent_scope": ["training", "text_training"],
                "access_class": access_class,
                "redistribution_permitted": redistribution_permitted,
                "training_permitted": True,
                "reviews": [
                    {
                        "reviewer_id": "TEST_REV_01",
                        "is_native_sl_krio_speaker": True,
                        "reviewed_on": "2024-01-02",
                        "verdict": "approve",
                    },
                    {
                        "reviewer_id": "TEST_REV_02",
                        "is_native_sl_krio_speaker": True,
                        "reviewed_on": "2024-01-02",
                        "verdict": "approve",
                    },
                ],
            },
            "text": "TEST_FIXTURE_STRICT_TEXT_001",
            "script": "Latin",
        }

    def test_strict_mode_raises_with_exact_line_number(self, tmp_path: Path, monkeypatch):
        """SLAI-022: One valid + one malformed line -> strict mode raises DataValidationError at line 2."""
        import sierraleone_ai.config as cfg_module

        krio_cleaned = tmp_path / "krio" / "cleaned"
        krio_cleaned.mkdir(parents=True)
        target = krio_cleaned / "krio_text_train.jsonl"
        line1 = json.dumps(self._valid_schema_dict())
        line2 = "{MALFORMED_JSON"
        target.write_text(f"{line1}\n{line2}\n", encoding="utf-8")

        monkeypatch.setattr(cfg_module.settings, "data_dir", tmp_path)

        with pytest.raises(DataValidationError) as exc_info:
            load_krio_text(split="train", strict=True)

        report = exc_info.value.report
        assert len(report.issues) == 1
        assert report.issues[0].line == 2
        assert "JSONDecodeError" in report.issues[0].error

    def test_preview_mode_returns_valid_records_and_writes_quarantine(
        self, tmp_path: Path, monkeypatch
    ):
        """SLAI-022: strict=False returns valid record + issue list + quarantine file."""
        import sierraleone_ai.config as cfg_module

        krio_cleaned = tmp_path / "krio" / "cleaned"
        krio_cleaned.mkdir(parents=True)
        target = krio_cleaned / "krio_text_train.jsonl"
        line1 = json.dumps(self._valid_schema_dict())
        bad_rec = self._valid_schema_dict(record_id="NOT-A-VALID-UUID")
        line2 = json.dumps(bad_rec)
        target.write_text(f"{line1}\n{line2}\n", encoding="utf-8")

        monkeypatch.setattr(cfg_module.settings, "data_dir", tmp_path)

        report = load_krio_text(split="train", strict=False, write_quarantine=True)
        assert len(report) == 1
        assert len(report.issues) == 1
        assert report.issues[0].line == 2
        quarantine_file = tmp_path / "krio" / "quarantine" / "krio_text_train.issues.jsonl"
        assert quarantine_file.exists()

    def test_unreviewed_source_in_register_blocked(self, tmp_path: Path, monkeypatch):
        """SLAI-031: Sources marked UNREVIEWED in datasets/source_register.csv are blocked."""
        rec = self._valid_schema_dict(source="jaward_krio_corpus")
        admitted, skipped = _apply_policy([rec], "train", LoadPolicy(), "test.jsonl")
        assert len(admitted) == 0
        assert any("unreviewed_source_rights" in k for k in skipped)

    def test_duplicate_record_id_blocked(self):
        """SLAI-028: Duplicate record_id values within a load are rejected."""
        rec1 = self._valid_schema_dict(record_id="22222222-2222-4222-8222-222222222222")
        rec2 = self._valid_schema_dict(record_id="22222222-2222-4222-8222-222222222222")
        admitted, skipped = _apply_policy([rec1, rec2], "train", LoadPolicy(), "test.jsonl")
        assert len(admitted) == 1
        assert skipped["duplicate_record_id"] == 1

    def test_scan_jsonl_for_pii_detects_phone_and_email(self, tmp_path: Path):
        """SLAI-025: PII scanner detects SL phone numbers and emails in JSONL."""
        path = tmp_path / "pii_sample.jsonl"
        _write_jsonl(
            path,
            [
                {"text": "Call +232 76 123456 or email user@example.org."},
                {"text": "Clean text with no personal identifiers."},
            ],
        )
        findings = scan_jsonl_for_pii(path)
        assert len(findings) >= 2
        errors = " ".join(f.error for f in findings)
        assert "email" in errors
        assert "phone" in errors

    def test_export_for_release_excludes_non_public_and_no_redistribution(self):
        """SLAI-025: Public release export filters out internal/restricted or non-redistributable records."""
        from sierraleone_ai.datasets.schema import KrioTextRecord

        pub_dict = self._valid_schema_dict(
            record_id="33333333-3333-4333-8333-333333333333",
            access_class="public",
            redistribution_permitted=True,
        )
        pub_dict["metadata"]["consent_scope"] = ["training", "public_release"]
        pub_rec = KrioTextRecord.model_validate(pub_dict)
        internal_rec = KrioTextRecord.model_validate(
            self._valid_schema_dict(
                record_id="44444444-4444-4444-8444-444444444444",
                access_class="internal",
                redistribution_permitted=True,
            )
        )
        no_redist_rec = KrioTextRecord.model_validate(
            self._valid_schema_dict(
                record_id="55555555-5555-4555-8555-555555555555",
                access_class="public",
                redistribution_permitted=False,
            )
        )
        exported = export_for_release([pub_rec, internal_rec, no_redist_rec])
        assert len(exported) == 1
        assert exported[0].metadata.record_id == "33333333-3333-4333-8333-333333333333"


