"""Tests for the FastAPI application endpoints."""
import io

from fastapi.testclient import TestClient

from sierraleone_ai.api.app import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert isinstance(data["models_operational"], list)
    assert data["models_operational"] == []


def test_health_live_returns_200():
    """SLAI-027: /health/live returns 200 ok when process is running."""
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_ready_returns_503_when_no_model_loaded():
    """SLAI-027: /health/ready returns 503 until at least one model is operational."""
    response = client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["detail"]["status"] == "not_ready"


def test_languages():
    response = client.get("/languages")
    assert response.status_code == 200
    langs = response.json()
    assert len(langs) == 8
    codes = [lang["code"] for lang in langs]
    for expected in ("krio", "mende", "temne", "limba", "loko", "kono", "kissi", "mandingo"):
        assert expected in codes
    # SLAI-027: All languages currently report planned/active, never operational
    for lang in langs:
        assert lang["status"] in ("planned", "active", "operational")
        assert lang["status"] != "operational"


def test_generate_returns_503():
    response = client.post(
        "/generate",
        json={"prompt": "FIXTURE_PROMPT_001", "language": "krio"},
    )
    assert response.status_code == 503
    assert response.json()["detail"]["error"] == "model_not_ready"


def test_generate_unsupported_language_returns_422():
    response = client.post(
        "/generate",
        json={"prompt": "FIXTURE_PROMPT_001", "language": "klingon"},
    )
    assert response.status_code == 422


def test_translate_returns_503():
    response = client.post(
        "/translate",
        json={"text": "Hello", "source_language": "english", "target_language": "krio"},
    )
    assert response.status_code == 503


def test_translate_unsupported_language_returns_422():
    response = client.post(
        "/translate",
        json={"text": "Hello", "source_language": "martian", "target_language": "krio"},
    )
    assert response.status_code == 422


def test_transcribe_returns_503():
    """Transcribe requires a multipart file upload."""
    fake_audio = io.BytesIO(b"RIFF....WAVEfmt ")
    response = client.post(
        "/transcribe",
        files={"file": ("test.wav", fake_audio, "audio/wav")},
    )
    assert response.status_code == 503


def test_synthesize_returns_503():
    response = client.post(
        "/synthesize",
        json={"text": "FIXTURE_TEXT_001", "language": "krio"},
    )
    assert response.status_code == 503


def test_synthesize_unsupported_language_returns_422():
    response = client.post(
        "/synthesize",
        json={"text": "FIXTURE_TEXT_001", "language": "klingon"},
    )
    assert response.status_code == 422


def test_transcribe_rejects_unsupported_content_type():
    """SLAI-027: Non-audio MIME type returns HTTP 415."""
    fake_file = io.BytesIO(b"plain text content")
    response = client.post(
        "/transcribe",
        files={"file": ("test.txt", fake_file, "text/plain")},
    )
    assert response.status_code == 415


def test_transcribe_rejects_empty_audio():
    """SLAI-027: Empty audio upload returns HTTP 400."""
    empty_audio = io.BytesIO(b"")
    response = client.post(
        "/transcribe",
        files={"file": ("empty.wav", empty_audio, "audio/wav")},
    )
    assert response.status_code == 400


def test_transcribe_rejects_malformed_audio_bytes():
    """SLAI-027: Non-audio header bytes disguised as audio/wav return HTTP 400."""
    bad_audio = io.BytesIO(b"NOT_A_VALID_AUDIO_HEADER_1234567890")
    response = client.post(
        "/transcribe",
        files={"file": ("bad.wav", bad_audio, "audio/wav")},
    )
    assert response.status_code == 400


def test_transcribe_rejects_oversized_upload(monkeypatch):
    """SLAI-027: Audio exceeding settings.max_audio_upload_bytes returns HTTP 413."""
    import sierraleone_ai.config as cfg_module

    monkeypatch.setattr(cfg_module.settings, "max_audio_upload_bytes", 16)
    oversized = io.BytesIO(b"RIFF....WAVEfmt " + b"0" * 32)
    response = client.post(
        "/transcribe",
        files={"file": ("big.wav", oversized, "audio/wav")},
    )
    assert response.status_code == 413


def test_translate_requires_english_on_exactly_one_side():
    """PA-012: /translate requires English on exactly one side (422 otherwise)."""
    # Neither side English -> 422
    resp = client.post(
        "/translate",
        json={"text": "Hello", "source_language": "krio", "target_language": "mende"},
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "invalid_translation_pair"

    # Both sides English -> 422
    resp = client.post(
        "/translate",
        json={"text": "Hello", "source_language": "english", "target_language": "english"},
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "invalid_translation_pair"


def test_generate_rejects_english():
    """PA-012: /generate is for Sierra Leonean languages and rejects English with 422."""
    resp = client.post(
        "/generate",
        json={"prompt": "FIXTURE_PROMPT_001", "language": "english"},
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "unsupported_language"


def test_health_ready_state_based():
    """PA-012: /health/ready returns 503 until model_state has loaded models, then 200."""
    from sierraleone_ai.serving.state import model_state

    model_state.reset()
    resp = client.get("/health/ready")
    assert resp.status_code == 503

    model_state.register_model("chat_llm", "krio-llm-v1")
    try:
        resp = client.get("/health/ready")
        assert resp.status_code == 200
        assert "krio-llm-v1" in resp.json()["models_operational"]
    finally:
        model_state.reset()

    resp = client.get("/health/ready")
    assert resp.status_code == 503


def test_transcribe_accepts_mp4_m4a_amr():
    """PA-012: /transcribe accepts m4a and amr magic prefixes (returns 503 model not ready)."""
    m4a_audio = io.BytesIO(b"\x00\x00\x00\x20ftypM4A ")
    resp = client.post(
        "/transcribe",
        files={"file": ("test.m4a", m4a_audio, "audio/x-m4a")},
    )
    assert resp.status_code == 503

    amr_audio = io.BytesIO(b"#!AMR\n" + b"\x00" * 16)
    resp = client.post(
        "/transcribe",
        files={"file": ("test.amr", amr_audio, "audio/amr")},
    )
    assert resp.status_code == 503


def test_transcribe_rejects_riff_non_wave():
    """PA-012: RIFF file that is not WAVE (e.g. AVI/WEBP) is rejected with 400."""
    non_wave = io.BytesIO(b"RIFF....AVI fmt ")
    resp = client.post(
        "/transcribe",
        files={"file": ("test.wav", non_wave, "audio/wav")},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["error"] == "malformed_audio"


