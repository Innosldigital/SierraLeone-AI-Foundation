"""
Tests for benchmark contamination fingerprinting and integrity (SLAI-007 / SLAI-017 / PA-004).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from sierraleone_ai.datasets.schema import BenchmarkItem
from sierraleone_ai.evaluation.benchmarks import (
    assert_held_out_integrity,
    build_benchmark_manifest,
    check_held_out_integrity,
    find_benchmark_overlap,
    iter_texts,
    text_fingerprint,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _benchmark_item_dict(
    item_id: str = "item-001",
    prompt: str = "Wetin mek yu de kam?",
    references: list[str] | None = None,
    review_count: int = 2,
    source: str = "test_fixture_approved_source",
    split: str = "held_out",
) -> dict:
    return {
        "metadata": {
            "record_id": "00000000-0000-4000-8000-000000000001",
            "language": "krio",
            "source": source,
            "collection_date": "2024-01-01",
            "licence": "CC-BY-4.0",
            "licence_id": "CC-BY-4.0",
            "validation_status": "validated",
            "dataset_version": "0.1.0",
            "split": split,
            "contributor_consent": True,
            "consent_scope": ["public_release"],
            "training_permitted": False,
            "redistribution_permitted": True,
            "reviews": [
                {
                    "reviewer_id": "REV_01",
                    "is_native_speaker_of_record_language": True,
                    "reviewed_on": "2024-01-02",
                    "verdict": "approve",
                },
                {
                    "reviewer_id": "REV_02",
                    "is_native_speaker_of_record_language": True,
                    "reviewed_on": "2024-01-02",
                    "verdict": "approve",
                },
            ],
        },
        "item_id": item_id,
        "evaluation_category": "qa",
        "prompt": prompt,
        "references": references or ["Mi na fɔ kam luk yu."],
        "review_count": review_count,
        "benchmark_version": "0.1.0",
    }


def _benchmark_record(source: str, text: str) -> dict:
    return {
        "metadata": {
            "record_id": "00000000-0000-0000-0000-000000000000",
            "language": "krio",
            "source": source,
            "collection_date": "2024-01-01",
            "licence": "CC-BY-4.0",
            "validation_status": "validated",
            "dataset_version": "0.1.0",
            "split": "held_out",
            "contributor_consent": True,
            "consent_scope": ["evaluation"],
        },
        "text": text,
    }


class TestTextFingerprint:
    def test_same_text_same_fingerprint(self):
        fp1 = text_fingerprint("Hello World")
        fp2 = text_fingerprint("Hello World")
        assert fp1 == fp2

    def test_case_insensitive(self):
        assert text_fingerprint("HELLO WORLD") == text_fingerprint("hello world")

    def test_punctuation_stripped(self):
        """Punctuation differences should not change the fingerprint."""
        assert text_fingerprint("Hello, World!") == text_fingerprint("Hello World")

    def test_different_texts_different_fingerprints(self):
        assert text_fingerprint("alpha beta") != text_fingerprint("gamma delta")


class TestBenchmarkOverlap:
    def test_overlap_detected(self, tmp_path: Path, monkeypatch):
        import sierraleone_ai.config as cfg_module

        benchmarks_dir = tmp_path / "benchmarks"
        monkeypatch.setattr(cfg_module.settings, "benchmarks_dir", benchmarks_dir)

        benchmark_text = "FIXTURE_BENCHMARK_SENTENCE_UNIQUE_42"
        _write_jsonl(
            benchmarks_dir / "krio" / "held_out.jsonl",
            [_benchmark_record("real_benchmark_2024", benchmark_text)],
        )

        candidates = ["unrelated text", benchmark_text, "another unrelated text"]
        overlap_indices = find_benchmark_overlap(candidates, language="krio")
        assert 1 in overlap_indices

    def test_no_overlap_when_clear(self, tmp_path: Path, monkeypatch):
        import sierraleone_ai.config as cfg_module

        benchmarks_dir = tmp_path / "benchmarks"
        monkeypatch.setattr(cfg_module.settings, "benchmarks_dir", benchmarks_dir)

        _write_jsonl(
            benchmarks_dir / "krio" / "held_out.jsonl",
            [_benchmark_record("real_benchmark_2024", "FIXTURE_UNIQUE_BENCHMARK_SENTENCE_XYZ")],
        )

        candidates = ["completely different", "nothing to see here"]
        overlap_indices = find_benchmark_overlap(candidates, language="krio")
        assert overlap_indices == []

    def test_case_and_punct_variants_detected(self, tmp_path: Path, monkeypatch):
        """Capitalisation and punctuation variants must still be detected as overlap."""
        import sierraleone_ai.config as cfg_module

        benchmarks_dir = tmp_path / "benchmarks"
        monkeypatch.setattr(cfg_module.settings, "benchmarks_dir", benchmarks_dir)

        _write_jsonl(
            benchmarks_dir / "krio" / "held_out.jsonl",
            [_benchmark_record("real_benchmark_2024", "fixture benchmark sentence")],
        )

        # Capitalised + punctuation variant
        candidates = ["FIXTURE BENCHMARK SENTENCE!"]
        overlap_indices = find_benchmark_overlap(candidates, language="krio")
        assert 0 in overlap_indices


class TestIterTexts:
    def test_iter_texts_extracts_all_text_bearing_fields(self):
        """PA-004: iter_texts yields text, prompt, references, questions, answers, turns."""
        doc = {
            "prompt": "prompt text",
            "references": ["ref1", "ref2"],
            "question": "q text",
            "answer": "a text",
            "turns": [{"text": "turn1"}, {"text": "turn2"}],
        }
        extracted = list(iter_texts(doc))
        assert "prompt text" in extracted
        assert "ref1" in extracted
        assert "ref2" in extracted
        assert "q text" in extracted
        assert "a text" in extracted
        assert "turn1" in extracted
        assert "turn2" in extracted


class TestBenchmarkItem:
    def test_valid_benchmark_item(self):
        data = _benchmark_item_dict()
        item = BenchmarkItem.model_validate(data)
        assert item.item_id == "item-001"
        assert item.review_count == 2

    def test_benchmark_item_rejects_non_held_out_split(self):
        data = _benchmark_item_dict(split="train")
        with pytest.raises(ValidationError, match="split=DataSplit.HELD_OUT"):
            BenchmarkItem.model_validate(data)


class TestAssertHeldOutIntegrity:
    def test_no_real_records_returns_false(self, tmp_path: Path, monkeypatch):
        import sierraleone_ai.config as cfg_module

        monkeypatch.setattr(cfg_module.settings, "benchmarks_dir", tmp_path / "benchmarks")

        result = assert_held_out_integrity(language="krio")
        assert result is False

    def test_template_only_returns_false(self, tmp_path: Path, monkeypatch):
        import sierraleone_ai.config as cfg_module

        benchmarks_dir = tmp_path / "benchmarks"
        monkeypatch.setattr(cfg_module.settings, "benchmarks_dir", benchmarks_dir)

        _write_jsonl(
            benchmarks_dir / "krio" / "held_out.jsonl",
            [_benchmark_record("BENCHMARK_SCHEMA_TEMPLATE_NOT_VALIDATED", "template text")],
        )

        result = assert_held_out_integrity(language="krio")
        assert result is False

    def test_manifest_and_integrity_check(self, tmp_path: Path, monkeypatch):
        """PA-004: build_benchmark_manifest and check_held_out_integrity."""
        import sierraleone_ai.config as cfg_module

        benchmarks_dir = tmp_path / "benchmarks"
        monkeypatch.setattr(cfg_module.settings, "benchmarks_dir", benchmarks_dir)
        monkeypatch.setattr(cfg_module.settings, "min_benchmark_items", 2)

        krio_bench = benchmarks_dir / "krio"
        items = [
            _benchmark_item_dict(item_id="item-001", prompt="Prompt 1"),
            _benchmark_item_dict(item_id="item-002", prompt="Prompt 2"),
        ]
        target = krio_bench / "benchmark_items.jsonl"
        _write_jsonl(target, items)

        manifest_path = build_benchmark_manifest("krio")
        assert manifest_path.exists()

        check = check_held_out_integrity("krio")
        assert check["ok"] is True
        assert check["count"] == 2
        assert check["reasons"] == []

        # Tampering with file fails integrity check
        target.write_text(target.read_text(encoding="utf-8") + "\n# extra\n", encoding="utf-8")
        stale_check = check_held_out_integrity("krio")
        assert stale_check["ok"] is False
        assert any("hash_mismatch" in r for r in stale_check["reasons"])

    def test_training_loader_drops_benchmark_prompt_overlap(self, tmp_path: Path, monkeypatch):
        """PA-004: loader drops training row that overlaps with benchmark prompt."""
        import sierraleone_ai.config as cfg_module
        from sierraleone_ai.datasets.loader import load_krio_text

        data_dir = tmp_path / "datasets"
        benchmarks_dir = tmp_path / "benchmarks"
        monkeypatch.setattr(cfg_module.settings, "data_dir", data_dir)
        monkeypatch.setattr(cfg_module.settings, "benchmarks_dir", benchmarks_dir)

        prompt_text = "kusheh, wetin mek yu kam na ya?"
        _write_jsonl(
            benchmarks_dir / "krio" / "eval_real.jsonl",
            [_benchmark_item_dict(prompt=prompt_text)],
        )

        # Training row with capitalisation & punctuation variation of the prompt
        training_row = {
            "metadata": {
                "record_id": "99999999-9999-4999-8999-999999999999",
                "language": "krio",
                "source": "test_fixture_approved_source",
                "collection_date": "2024-01-01",
                "licence": "CC-BY-4.0",
                "licence_id": "CC-BY-4.0",
                "validation_status": "validated",
                "dataset_version": "0.1.0",
                "split": "train",
                "contributor_consent": True,
                "consent_scope": ["text_training"],
                "access_class": "public",
                "redistribution_permitted": True,
                "training_permitted": True,
                "reviews": [
                    {
                        "reviewer_id": "REV_01",
                        "is_native_speaker_of_record_language": True,
                        "reviewed_on": "2024-01-02",
                        "verdict": "approve",
                    },
                    {
                        "reviewer_id": "REV_02",
                        "is_native_speaker_of_record_language": True,
                        "reviewed_on": "2024-01-02",
                        "verdict": "approve",
                    },
                ],
            },
            "text": "KUSHEH WETIN MEK YU KAM NA YA",
            "script": "Latin",
        }
        cleaned_dir = data_dir / "krio" / "cleaned"
        _write_jsonl(cleaned_dir / "krio_text_train.jsonl", [training_row])

        report = load_krio_text(split="train")
        assert len(report.records) == 0
        assert report.skipped.get("benchmark_overlap", 0) == 1
