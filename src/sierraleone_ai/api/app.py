"""
Sierra Leone AI Foundation – FastAPI application.

IMPORTANT: All model endpoints are STUBS.
No model is currently trained or operational.
Endpoints return informative HTTP 503 responses until models are available.
"""
from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from sierraleone_ai import SUPPORTED_LANGUAGES, __version__
from sierraleone_ai.api.models import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    LanguageInfo,
    SynthesizeRequest,
    SynthesizeResponse,
    TranscribeResponse,
    TranslateRequest,
    TranslateResponse,
)
from sierraleone_ai.config import settings

app = FastAPI(
    title="Sierra Leone AI Foundation API",
    description=(
        "API for Sierra Leonean language AI services.\n\n"
        "**Status**: Foundation phase. Model endpoints are stubs – "
        "no production model is currently operational."
    ),
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins or ["*"],  # restrict in production via SLAI_CORS_ORIGINS
    allow_methods=["*"],
    allow_headers=["*"],
)

# Valid language codes accepted by model endpoints (lowercase strings)
_VALID_LANG_CODES = set(SUPPORTED_LANGUAGES.keys()) | {"english"}


def _model_not_ready() -> HTTPException:
    """Return a fresh HTTPException for each raise so the instance is not shared."""
    return HTTPException(
        status_code=503,
        detail={
            "error": "model_not_ready",
            "message": (
                "No language model has been trained yet. "
                "This endpoint will become operational after the Krio language model "
                "has been trained, evaluated and approved for deployment. "
                "See DEVELOPMENT_ROADMAP.md for the current status."
            ),
        },
    )


def _validate_language(language: str) -> None:
    """Raise 422 if the language code is not in SUPPORTED_LANGUAGES."""
    if language.lower() not in _VALID_LANG_CODES:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "unsupported_language",
                "message": (
                    f"Language '{language}' is not supported. "
                    f"Supported codes: {sorted(_VALID_LANG_CODES)}"
                ),
            },
        )


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    """System health check."""
    return HealthResponse(
        status="ok",
        version=__version__,
        models_operational=[],
        note="Foundation phase – no models are operational yet.",
    )


@app.get("/languages", response_model=list[LanguageInfo], tags=["System"])
async def languages() -> list[LanguageInfo]:
    """List all supported and planned languages."""
    return [
        LanguageInfo(
            code=code,
            name=info["name"],
            iso_639_3=info.get("iso_639_3", ""),
            status=info["status"],
        )
        for code, info in SUPPORTED_LANGUAGES.items()
    ]


@app.post(
    "/generate",
    response_model=GenerateResponse,
    tags=["Language Model"],
    responses={503: {"description": "Model not yet operational"}},
)
async def generate(request: GenerateRequest) -> GenerateResponse:
    """
    Generate text in a Sierra Leonean language.

    **Status**: Not yet operational. Returns 503 until a model is trained.
    """
    _validate_language(request.language)
    raise _model_not_ready()


@app.post(
    "/translate",
    response_model=TranslateResponse,
    tags=["Translation"],
    responses={503: {"description": "Model not yet operational"}},
)
async def translate(request: TranslateRequest) -> TranslateResponse:
    """
    Translate text between English and a Sierra Leonean language.

    **Status**: Not yet operational. Returns 503 until a translation model is trained.
    """
    _validate_language(request.source_language)
    _validate_language(request.target_language)
    raise _model_not_ready()


@app.post(
    "/transcribe",
    tags=["Speech Recognition"],
    responses={503: {"description": "Model not yet operational"}},
)
async def transcribe(
    file: UploadFile = File(..., description="Audio file to transcribe (WAV/MP3/FLAC)"),
) -> TranscribeResponse:
    """
    Transcribe audio to text in a Sierra Leonean language.

    Accepts a multipart/form-data audio file upload.

    **Status**: Not yet operational. Returns 503 until an ASR model is trained.
    """
    raise _model_not_ready()


@app.post(
    "/synthesize",
    tags=["Text-to-Speech"],
    responses={503: {"description": "Model not yet operational"}},
)
async def synthesize(request: SynthesizeRequest) -> SynthesizeResponse:
    """
    Synthesize speech from text in a Sierra Leonean language.

    **Status**: Not yet operational. Returns 503 until a TTS model is trained.
    """
    _validate_language(request.language)
    raise _model_not_ready()
