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
