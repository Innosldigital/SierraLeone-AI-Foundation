"""
Global configuration for the Sierra Leone AI Foundation.
Uses pydantic-settings so values can be overridden via environment variables or .env.

Environment variable prefix: SLAI_
Exception: HF_TOKEN is also accepted without the prefix (standard HuggingFace convention).
"""
from __future__ import annotations

import os
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Project root — resolved relative to this file so the package works
# regardless of the current working directory (editable installs, scripts, etc.)
# Override with SLAI_PROJECT_ROOT=/absolute/path if needed.
# ---------------------------------------------------------------------------
_DEFAULT_ROOT = Path(os.environ.get(
    "SLAI_PROJECT_ROOT",
    str(Path(__file__).resolve().parents[2]),
))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SLAI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Project root (all other paths are relative to this)
    project_root: Path = _DEFAULT_ROOT

    # Data directories (derived from project_root; override via env if needed)
    data_dir: Path = Field(default=_DEFAULT_ROOT / "datasets")
    models_dir: Path = Field(default=_DEFAULT_ROOT / "models")
    benchmarks_dir: Path = Field(default=_DEFAULT_ROOT / "benchmarks")
    logs_dir: Path = Field(default=_DEFAULT_ROOT / "logs")

    # API
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_debug: bool = False
    cors_origins: list[str] = Field(default_factory=list)
    max_audio_upload_bytes: int = 10 * 1024 * 1024  # 10 MB default upload cap (SLAI-027)

    # Safety gate – prevents accidental expensive training
    allow_training: bool = False

    # HuggingFace token — stored as SecretStr so it never appears in repr/logs.
    # Accepts both HF_TOKEN (standard HuggingFace convention) and SLAI_HF_TOKEN.
    hf_token: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("HF_TOKEN", "SLAI_HF_TOKEN"),
    )

    def language_data_dir(self, language_code: str) -> Path:
        return self.data_dir / language_code

    def language_benchmark_dir(self, language_code: str) -> Path:
        return self.benchmarks_dir / language_code

    # Convenience properties kept for backward compatibility
    @property
    def krio_data_dir(self) -> Path:
        return self.language_data_dir("krio")

    @property
    def mende_data_dir(self) -> Path:
        return self.language_data_dir("mende")

    @property
    def temne_data_dir(self) -> Path:
        return self.language_data_dir("temne")


settings = Settings()
