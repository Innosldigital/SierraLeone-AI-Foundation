"""
Sierra Leone AI Foundation – FastAPI application (SLAI-012 / SLAI-027).

IMPORTANT: All model endpoints are STUBS.
No model is currently trained or operational.
Endpoints return informative HTTP 503 responses until models are available.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from sierraleone_ai import (
    LANGUAGE_REGISTRY,
    SUPPORTED_LANGUAGES,
    __version__,
    get_language_state,
)
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
    allow_origins=settings.cors_origins or ["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_ALLOWED_AUDIO_CONTENT_TYPES: frozenset[str] = frozenset({
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/mpeg",
    "audio/mp3",
    "audio/ogg",
    "audio/flac",
    "audio/webm",
    "application/octet-stream",
})

_AUDIO_MAGIC_PREFIXES: tuple[bytes, ...] = (
    b"RIFF",              # WAV
    b"ID3",               # MP3 (ID3v2)
    b"\xff\xfb",          # MP3 frame sync
    b"\xff\xf3",          # MP3 frame sync
    b"\xff\xf2",          # MP3 frame sync
    b"fLaC",              # FLAC
    b"OggS",              # OGG
    b"\x1a\x45\xdf\xa3",  # WebM / Matroska
)


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


def _validate_and_require_operational_language(language: str) -> None:
    """
    Validate language against the single LANGUAGE_REGISTRY (SLAI-027 / SLAI-028):
      - Unknown language code -> 422
      - Non-operational language ('planned' or 'active') -> 503
    """
    code = language.lower()
    if code not in LANGUAGE_REGISTRY:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "unsupported_language",
                "message": (
                    f"Language '{language}' is not supported. "
                    f"Supported codes: {sorted(LANGUAGE_REGISTRY.keys())}"
                ),
            },
        )
    state = get_language_state(code)
    if state != "operational" and code != "english":
        raise _model_not_ready()


@app.get("/health/live", response_model=HealthResponse, tags=["System"])
async def health_live() -> HealthResponse:
    """Liveness check: confirms the API process is running (SLAI-027)."""
    return HealthResponse(
        status="ok",
        version=__version__,
        models_operational=[],
        note="Process is live. Foundation phase – no models are operational yet.",
    )


@app.get(
    "/health/ready",
    response_model=HealthResponse,
    tags=["System"],
    responses={503: {"description": "No operational models loaded"}},
)
async def health_ready() -> HealthResponse:
    """
    Readiness check: returns 200 only when at least one model is operational,
    otherwise returns 503 Service Unavailable (SLAI-027).
    """
    operational = [
        code for code, info in SUPPORTED_LANGUAGES.items()
        if info.get("status") == "operational"
    ]
    if not operational:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "model_not_ready",
                "status": "not_ready",
                "version": __version__,
                "models_operational": [],
                "message": "API process is live, but no trained models are loaded yet.",
            },
        )
    return HealthResponse(
        status="ready",
        version=__version__,
        models_operational=operational,
        note="Operational models loaded.",
    )


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    """Backward-compatible alias of `/health/live` (SLAI-027)."""
    return await health_live()


@app.get("/languages", response_model=list[LanguageInfo], tags=["System"])
async def languages() -> list[LanguageInfo]:
    """List all supported and planned Sierra Leonean languages."""
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
    _validate_and_require_operational_language(request.language)
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
    if request.source_language.lower() not in LANGUAGE_REGISTRY:
        _validate_and_require_operational_language(request.source_language)
    if request.target_language.lower() not in LANGUAGE_REGISTRY:
        _validate_and_require_operational_language(request.target_language)
    _validate_and_require_operational_language(request.source_language)
    _validate_and_require_operational_language(request.target_language)
    raise _model_not_ready()


@app.post(
    "/transcribe",
    tags=["Speech Recognition"],
    responses={
        400: {"description": "Empty or malformed audio upload"},
        413: {"description": "Audio upload exceeds maximum allowed size"},
        415: {"description": "Unsupported audio content type"},
        503: {"description": "Model not yet operational"},
    },
)
async def transcribe(
    file: UploadFile = File(..., description="Audio file to transcribe (WAV/MP3/FLAC)"),
) -> TranscribeResponse:
    """
    Transcribe audio to text in a Sierra Leonean language (SLAI-027).

    Validates content-type, size cap, and audio header, writes to a secure temporary
    file with guaranteed cleanup, and returns 503 until an ASR model is operational.
    """
    content_type = (file.content_type or "").split(";")[0].strip().lower()
    if content_type not in _ALLOWED_AUDIO_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail={
                "error": "unsupported_media_type",
                "message": f"Content-Type '{file.content_type}' is not a supported audio format.",
            },
        )

    max_bytes = settings.max_audio_upload_bytes
    payload = await file.read(max_bytes + 1)
    if len(payload) == 0:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "empty_audio_file",
                "message": "Uploaded audio file is empty (0 bytes).",
            },
        )
    if len(payload) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "audio_file_too_large",
                "message": f"Uploaded audio exceeds maximum allowed size ({max_bytes} bytes).",
            },
        )
    if not payload.startswith(_AUDIO_MAGIC_PREFIXES):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "malformed_audio",
                "message": "Uploaded file does not have a recognised audio header (WAV/MP3/FLAC/OGG/WebM).",
            },
        )

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as tmp:
            tmp.write(payload)
            tmp_path = Path(tmp.name)
        raise _model_not_ready()
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


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
    _validate_and_require_operational_language(request.language.value)
    raise _model_not_ready()
