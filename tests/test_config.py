"""
Tests for config loading (SLAI-006 / SLAI-017).

Verifies that:
- HF_TOKEN env var is loaded into settings.hf_token (via alias).
- The token value does not appear in repr(settings) or str(settings).
- Paths resolve to absolute locations.
"""

from __future__ import annotations


class TestConfigSecrets:
    def test_hf_token_loaded_via_hf_token_alias(self, monkeypatch):
        """HF_TOKEN env var must be loaded via AliasChoices."""
        monkeypatch.setenv("HF_TOKEN", "test-token-abc123")
        # Instantiate a fresh Settings object to pick up the patched env
        from sierraleone_ai.config import Settings

        settings = Settings()
        assert settings.hf_token is not None
        assert settings.hf_token.get_secret_value() == "test-token-abc123"

    def test_slai_hf_token_alias_also_works(self, monkeypatch):
        """SLAI_HF_TOKEN env var must also be accepted."""
        monkeypatch.setenv("SLAI_HF_TOKEN", "another-secret-token")
        from sierraleone_ai.config import Settings

        settings = Settings()
        assert settings.hf_token.get_secret_value() == "another-secret-token"

    def test_hf_token_not_in_repr(self, monkeypatch):
        """The raw token value must not leak into repr() or str()."""
        monkeypatch.setenv("SLAI_HF_TOKEN", "super-secret-token-xyz")
        from sierraleone_ai.config import Settings

        settings = Settings()
        assert "super-secret-token-xyz" not in repr(settings)
        assert "super-secret-token-xyz" not in str(settings)

    def test_data_dir_is_absolute(self):
        from sierraleone_ai.config import settings

        assert settings.data_dir.is_absolute(), (
            f"settings.data_dir must be absolute, got {settings.data_dir}"
        )

    def test_krio_data_dir_under_data_dir(self):
        from sierraleone_ai.config import settings

        assert settings.krio_data_dir.is_absolute()
        assert "krio" in str(settings.krio_data_dir)

    def test_allow_training_defaults_false(self):
        from sierraleone_ai.config import settings

        assert settings.allow_training is False

    def test_pyproject_optional_dependencies_uncommented(self):
        """SLAI-015: [project.optional-dependencies] groups training, audio, eval, data, dev are non-empty."""
        from pathlib import Path

        pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
        try:
            import tomllib

            data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
            opt_deps = data["project"]["optional-dependencies"]
            for group in ("training", "audio", "eval", "data", "dev"):
                assert group in opt_deps, f"Missing optional-dependencies group '{group}'"
                assert len(opt_deps[group]) >= 1, f"Optional dependency group '{group}' is empty"
        except ModuleNotFoundError:
            # Fallback for Python 3.10 where tomllib is not in the standard library
            text = pyproject_path.read_text(encoding="utf-8")
            assert "[project.optional-dependencies]" in text
            for group in ("training", "audio", "eval", "data", "dev"):
                assert f"{group} = [" in text, f"Missing optional-dependencies group '{group}'"

    def test_language_registry_matches_schema_enum_and_benchmark_dirs(self):
        """SLAI-028 / SLAI-018: Single language registry matches Language enum and lowercase benchmark dirs."""
        from pathlib import Path

        from sierraleone_ai import LANGUAGE_REGISTRY, SUPPORTED_LANGUAGES
        from sierraleone_ai.datasets.schema import Language

        enum_values = {lang.value for lang in Language if lang.value != "other"}
        assert set(LANGUAGE_REGISTRY.keys()) == enum_values
        assert set(SUPPORTED_LANGUAGES) == enum_values - {"english"}

        benchmarks_root = Path(__file__).resolve().parents[1] / "benchmarks"
        if benchmarks_root.exists():
            for child in benchmarks_root.iterdir():
                if child.is_dir():
                    assert child.name == child.name.lower(), (
                        f"Benchmark directory '{child.name}' must be lowercase"
                    )

    def test_no_template_file_marked_validated_in_repo(self):
        """SLAI-017: No file under datasets/ or benchmarks/ may have validation_status=='validated' with SCHEMA_TEMPLATE source."""
        import json
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[1]
        for folder in ("datasets", "benchmarks"):
            root = repo_root / folder
            if not root.exists():
                continue
            for jsonl_file in root.rglob("*.jsonl"):
                for lineno, line in enumerate(
                    jsonl_file.read_text(encoding="utf-8").splitlines(), start=1
                ):
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#"):
                        continue
                    rec = json.loads(stripped)
                    meta = rec.get("metadata", rec)
                    source = str(meta.get("source", ""))
                    status = str(meta.get("validation_status", ""))
                    if "SCHEMA_TEMPLATE" in source or "TEMPLATE_NOT" in source:
                        assert status != "validated", (
                            f"{jsonl_file}:{lineno} has template source '{source}' "
                            f"but validation_status='validated'"
                        )

    def test_relative_data_dir_resolves_against_project_root(self, monkeypatch):
        """PA-014: Overridden relative data_dir resolves against project_root."""
        monkeypatch.setenv("SLAI_DATA_DIR", "my_custom_datasets")
        from sierraleone_ai.config import Settings

        s = Settings()
        assert s.data_dir.is_absolute()
        assert str(s.data_dir).endswith("my_custom_datasets")

    def test_cors_origins_parsed_from_json(self, monkeypatch):
        """PA-014: SLAI_CORS_ORIGINS parses JSON list."""
        monkeypatch.setenv(
            "SLAI_CORS_ORIGINS", '["https://example.com", "https://app.example.com"]'
        )
        from sierraleone_ai.config import Settings

        s = Settings()
        assert s.cors_origins == ["https://example.com", "https://app.example.com"]

    def test_cors_origins_empty_string(self, monkeypatch):
        """PA-014: SLAI_CORS_ORIGINS empty string yields empty list."""
        monkeypatch.setenv("SLAI_CORS_ORIGINS", "")
        from sierraleone_ai.config import Settings

        s = Settings()
        assert s.cors_origins == []
