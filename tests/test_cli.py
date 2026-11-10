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

    def test_status_exits_zero(self):
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0, f"status command failed: {result.output}"

    def test_data_summary_exits_zero(self):
        result = runner.invoke(app, ["data-summary"])
        assert result.exit_code == 0, f"data-summary command failed: {result.output}"

    def test_check_readiness_exits_zero(self):
        result = runner.invoke(app, ["check-readiness", "krio"])
        assert result.exit_code == 0, f"check-readiness command failed: {result.output}"

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
