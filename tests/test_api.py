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

