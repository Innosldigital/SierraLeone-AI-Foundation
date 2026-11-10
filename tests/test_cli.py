"""
Tests for the CLI commands (SLAI-017 / SLAI-020).

Uses typer's CliRunner to invoke commands without a live process.
"""
from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from sierraleone_ai.cli import app

runner = CliRunner()


class TestCLI:
    def test_version_exits_zero(self):
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0, f"version command failed: {result.output}"
        assert "0.1.0" in result.output

    def test_languages_exits_zero(self):
        result = runner.invoke(app, ["languages"])
        assert result.exit_code == 0, f"languages command failed: {result.output}"
        assert "krio" in result.output.lower()

    def test_status_exits_zero_and_uses_five_canonical_levels(self):
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0, f"status command failed: {result.output}"
        for level in ("implemented", "scaffolded", "not configured", "not ready"):
            assert level in result.output

    def test_data_summary_exits_zero_and_marked_preview(self):
        result = runner.invoke(app, ["data-summary"])
        assert result.exit_code == 0, f"data-summary command failed: {result.output}"
        assert "PREVIEW" in result.output

    def test_check_readiness_exits_zero(self):
        result = runner.invoke(app, ["check-readiness", "krio"])
        assert result.exit_code == 0, f"check-readiness command failed: {result.output}"

    def test_check_readiness_dry_run_exits_zero(self):
        result = runner.invoke(app, ["check-readiness", "krio", "--dry-run"])
        assert result.exit_code == 0, f"check-readiness --dry-run failed: {result.output}"
        assert "dry_run" in result.output or "Dry-Run" in result.output

    def test_audit_tokenizer_refuses_template_file(self, tmp_path: Path):
        """SLAI-029: audit-tokenizer refuses example_* and template files."""
        tmpl = tmp_path / "example_krio_text.jsonl"
        tmpl.write_text('{"text": "TEST_FIXTURE"}\n', encoding="utf-8")
        result = runner.invoke(
            app,
            [
                "audit-tokenizer",
                "--corpus",
                str(tmpl),
                "--tokenizer",
                "Qwen/Qwen3-4B",
            ],
        )
        assert result.exit_code == 1
        assert "Refusing to run tokenizer audit" in result.output

    def test_audit_tokenizer_with_mock_tokenizer_and_valid_corpus(
        self, tmp_path: Path, monkeypatch
    ):
        """SLAI-029: audit_tokenizer_on_corpus computes fertility, char/token, UNK rate, open vowel handling."""
        import json

        from sierraleone_ai.evaluation.tokenizer_audit import audit_tokenizer_on_corpus

        corpus = tmp_path / "validated_corpus.jsonl"
        rec = {
            "metadata": {
                "record_id": "99999999-9999-4999-8999-999999999999",
                "language": "krio",
                "source": "community_approved_2024",
                "collection_date": "2024-01-01",
                "licence": "CC-BY-4.0",
                "licence_id": "CC-BY-4.0",
                "validation_status": "validated",
                "dataset_version": "0.1.0",
                "split": "train",
                "contributor_consent": True,
                "consent_scope": ["training", "text_training"],
                "reviews": [
                    {
                        "reviewer_id": "REV_01",
                        "is_native_sl_krio_speaker": True,
                        "reviewed_on": "2024-01-02",
                        "verdict": "approve",
                    },
                    {
                        "reviewer_id": "REV_02",
                        "is_native_sl_krio_speaker": True,
                        "reviewed_on": "2024-01-02",
                        "verdict": "approve",
                    },
                ],
            },
            "text": "TEST_FIXTURE_WORD_ONE TEST_FIXTURE_WORD_TWO ɛ ɔ",
            "script": "Latin",
        }
        corpus.write_text(json.dumps(rec) + "\n", encoding="utf-8")

        class _DummyTokenizer:
            unk_token_id: int | None = 0

            def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
                return [i + 1 for i, _ in enumerate(text.split())]

            def tokenize(self, text: str) -> list[str]:
                return list(text)

        report = audit_tokenizer_on_corpus(
            corpus_path=corpus,
            tokenizer=_DummyTokenizer(),
            tokenizer_id="dummy/model-a",
            min_sentences=1,
        )
        assert report.tokens_per_word == 1.0
        assert report.unknown_token_rate == 0.0
        assert "ɛ" in report.open_vowel_handling
        assert "ɔ" in report.open_vowel_handling

    def test_validate_file_valid(self, tmp_path: Path):
        test_file = tmp_path / "valid.jsonl"
        test_file.write_text('{"metadata": {"valid": true}, "text": "sample"}\n', encoding="utf-8")
        result = runner.invoke(app, ["validate-file", str(test_file)])
        assert result.exit_code == 0
        assert "Valid lines: 1" in result.output

    def test_validate_file_not_found(self):
        result = runner.invoke(app, ["validate-file", "nonexistent_file_path.jsonl"])
        assert result.exit_code == 1

    def test_no_args_shows_help(self):
        result = runner.invoke(app, [])
        # With no_args_is_help=True Typer/Click displays help (exit code 0 or 2 depending on Click version)
        assert result.exit_code in (0, 2)
        assert "Usage" in result.output or "Commands" in result.output or "Options" in result.output
