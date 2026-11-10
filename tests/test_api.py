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


def test_languages():
    response = client.get("/languages")
    assert response.status_code == 200
    langs = response.json()
    assert len(langs) >= 7
    codes = [lang["code"] for lang in langs]
    for expected in ("krio", "mende", "temne", "limba", "loko", "kono", "kissi"):
        assert expected in codes


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
