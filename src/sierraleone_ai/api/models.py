"""Pydantic models for the API request/response layer."""
from __future__ import annotations

from pydantic import BaseModel, Field

from sierraleone_ai import SUPPORTED_LANGUAGES

# Valid language codes for field-level documentation
_LANG_CODES = sorted(SUPPORTED_LANGUAGES.keys())


class HealthResponse(BaseModel):
    status: str
    version: str
    models_operational: list[str]
    note: str


class LanguageInfo(BaseModel):
    code: str
    name: str
    iso_639_3: str
    status: str


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4096)
    language: str = Field(
        "krio",
        description=f"Target language code. Supported: {_LANG_CODES}",
    )
    max_new_tokens: int = Field(256, ge=1, le=2048)
    temperature: float = Field(0.7, ge=0.0, le=2.0)


class GenerateResponse(BaseModel):
    generated_text: str
    language: str
    model_id: str
    tokens_generated: int


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4096)
    source_language: str = Field(
        ..., description=f"Source language code. Supported: {_LANG_CODES + ['english']}"
    )
    target_language: str = Field(
        ..., description=f"Target language code. Supported: {_LANG_CODES + ['english']}"
    )


class TranslateResponse(BaseModel):
    translated_text: str
    source_language: str
    target_language: str
    model_id: str


class TranscribeResponse(BaseModel):
    transcript: str
    language: str
    confidence: float | None = None
    model_id: str


class SynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4096)
    language: str = Field(
        "krio",
        description=f"Target language code for speech synthesis. Supported: {_LANG_CODES}",
    )


class SynthesizeResponse(BaseModel):
    audio_url: str
    language: str
    duration_seconds: float
    model_id: str
