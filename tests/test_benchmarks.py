"""
Tests for benchmark contamination fingerprinting (SLAI-007 / SLAI-017).
"""
from __future__ import annotations

import json
from pathlib import Path

from sierraleone_ai.evaluation.benchmarks import (
    assert_held_out_integrity,
    find_benchmark_overlap,
    text_fingerprint,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


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

