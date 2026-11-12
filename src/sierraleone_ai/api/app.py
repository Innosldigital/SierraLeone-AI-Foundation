"""
Sierra Leone AI Foundation – FastAPI application (SLAI-012 / SLAI-027 / PA-012 / PA-013).

IMPORTANT: All model endpoints are STUBS.
No model is currently trained or operational.
Endpoints return informative HTTP 503 responses until models are available.
"""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from sierraleone_ai import (
    LANGUAGE_REGISTRY,
    SUPPORTED_LANGUAGES,
    __version__,
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
from sierraleone_ai.serving.state import model_state

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

# CORS configuration (PA-012 / PA-014):
# Default empty list means no CORS unless configured or in debug mode.
_cors_origins = settings.cors_origins or (
    ["http://localhost:3000"] if settings.api_debug else []
)
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.middleware("http")
async def check_content_length_limit(request: Request, call_next: Any) -> Response:
    """Early size cap check on incoming Content-Length header (PA-012)."""
    cl = request.headers.get("content-length")
    if cl is not None:
        try:
            if int(cl) > settings.max_audio_upload_bytes:
                return JSONResponse(
                    status_code=413,
                    content={
                        "detail": {
                            "error": "audio_file_too_large",
                            "message": (
                                f"Request body ({cl} bytes) exceeds maximum allowed size "
                                f"({settings.max_audio_upload_bytes} bytes)."
                            ),
                        }
                    },
                )
        except ValueError:
            pass
    return await call_next(request)


_ALLOWED_AUDIO_CONTENT_TYPES: frozenset[str] = frozenset({
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/mpeg",
    "audio/mp3",
    "audio/ogg",
    "audio/flac",
    "audio/webm",
    "audio/mp4",
    "audio/x-m4a",
    "audio/aac",
    "audio/3gpp",
    "audio/amr",
    "application/octet-stream",
})


def _is_valid_audio_header(payload: bytes) -> bool:
    """Validate audio header bytes against known magic signatures (PA-012)."""
    if len(payload) < 2:
        return False
    # WAV check: RIFF....WAVE
    if len(payload) >= 12 and payload[:4] == b"RIFF" and payload[8:12] == b"WAVE":
        return True
    # MP3 ID3v2 or frame sync
    if payload.startswith(b"ID3"):
        return True
    if payload.startswith((b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")):
        return True
    # FLAC
    if payload.startswith(b"fLaC"):
        return True
    # OGG
    if payload.startswith(b"OggS"):
        return True
    # WebM / Matroska
    if payload.startswith(b"\x1a\x45\xdf\xa3"):
        return True
    # MP4 / M4A / 3GP (ftyp box at offset 4)
    if len(payload) >= 8 and payload[4:8] == b"ftyp":
        return True
    # AMR
    if payload.startswith(b"#!AMR"):
        return True
    # ADTS AAC (syncword: 12 bits set to 1 -> 0xFFF...)
    if payload[0] == 0xFF and (payload[1] & 0xF0 == 0xF0):
        return True
    return False


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


def _require_language(code: str, *, allow_english: bool = True) -> str:
    """
    Validate language against the single LANGUAGE_REGISTRY once (PA-012):
      - Unknown language code -> 422
      - English when not allowed -> 422
    """
    c = code.lower().strip()
    if c not in LANGUAGE_REGISTRY or (c == "english" and not allow_english):
        valid = sorted(k for k in LANGUAGE_REGISTRY if allow_english or k != "english")
        raise HTTPException(
            status_code=422,
            detail={
                "error": "unsupported_language",
                "message": f"Language '{code}' is not supported. Supported codes: {valid}",
            },
        )
    return c


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
    otherwise returns 503 Service Unavailable (SLAI-027 / PA-012).
    """
    if not model_state.is_ready():
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
    operational = model_state.get_loaded_models()
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
    _require_language(request.language, allow_english=False)
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
    src = _require_language(request.source_language, allow_english=True)
    tgt = _require_language(request.target_language, allow_english=True)

    # PA-012: require English on exactly one side
    if (src == "english") == (tgt == "english"):
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_translation_pair",
                "message": "Translation requires English on exactly one side.",
            },
        )

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
    file: UploadFile = File(..., description="Audio file to transcribe"),
) -> TranscribeResponse:
    """
    Transcribe audio to text in a Sierra Leonean language (SLAI-027 / PA-012).

    Validates content-type, size cap, and audio header, and returns 503 until an ASR model is operational.
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
    if not _is_valid_audio_header(payload):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "malformed_audio",
                "message": "Uploaded file does not have a recognised audio header (WAV/MP3/FLAC/OGG/WebM/MP4/M4A/AMR/AAC).",
            },
        )

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
    lang_str = request.language.value if hasattr(request.language, "value") else str(request.language)
    _require_language(lang_str, allow_english=False)
    raise _model_not_ready()
