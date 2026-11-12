# API Documentation

Base URL: `http://127.0.0.1:8000` (development)

Interactive docs: `http://127.0.0.1:8000/docs` (Swagger UI) and `/redoc`

> ⚠️ All model endpoints currently return HTTP 503 Service Unavailable. No language model, ASR, or TTS model is operational yet.

---

## Health & System Endpoints

### GET /health/live
Liveness probe: returns HTTP 200 `ok` as long as the process is alive.

**Response (200 OK)**
```json
{
  "status": "ok",
  "version": "0.1.0",
  "models_operational": [],
  "note": "Process is live. Foundation phase – no models are operational yet."
}
```

### GET /health/ready
Readiness probe: returns HTTP 200 only when at least one trained model is loaded into `ModelState`. Returns HTTP 503 otherwise.

**Response (503 Service Unavailable)**
```json
{
  "detail": {
    "error": "model_not_ready",
    "status": "not_ready",
    "version": "0.1.0",
    "models_operational": [],
    "message": "API process is live, but no trained models are loaded yet."
  }
}
```

### GET /health
Backward-compatible alias of `/health/live`.

### GET /languages
Lists all supported and planned Sierra Leonean languages from `LANGUAGE_REGISTRY`.

**Response (200 OK)**
```json
[
  {
    "code": "krio",
    "name": "Krio",
    "iso_639_3": "kri",
    "status": "active"
  },
  {
    "code": "mende",
    "name": "Mende",
    "iso_639_3": "men",
    "status": "planned"
  }
]
```

---

## Model Endpoints (503 Stubs)

### POST /generate *(stub – 503)*

Generate text in a Sierra Leonean language.

**Request**
```json
{
  "prompt": "TEST_FIXTURE_PROMPT_001",
  "language": "krio",
  "max_new_tokens": 256,
  "temperature": 0.7
}
```

**Errors**
- `422 Unprocessable Entity`: Language code not in registry, or `"english"` (generation is for local languages).
- `503 Service Unavailable`: Model not yet operational.

---

### POST /translate *(stub – 503)*

Translate text between English and a Sierra Leonean language.
Requires English on **exactly one side** (source or target).

**Request**
```json
{
  "text": "Hello, how are you?",
  "source_language": "english",
  "target_language": "krio"
}
```

**Errors**
- `422 Unprocessable Entity`: Language code not supported, or English not on exactly one side (e.g. Krio-to-Mende or English-to-English).
- `503 Service Unavailable`: Model not yet operational.

---

### POST /transcribe *(stub – 503)*

Transcribe audio to text. Accepts multipart/form-data upload.

**Accepted Content Types:**
- `audio/wav`, `audio/mpeg`, `audio/mp3`, `audio/ogg`, `audio/flac`, `audio/webm`, `audio/mp4`, `audio/x-m4a`, `audio/aac`, `audio/3gpp`, `audio/amr`

**Size Limit:** Default 10 MB (10,485,760 bytes), configured via `SLAI_MAX_AUDIO_UPLOAD_BYTES`.

**Errors**
- `400 Bad Request`: Empty upload (0 bytes) or unrecognized audio magic header.
- `413 Payload Too Large`: Upload exceeds maximum allowed bytes.
- `415 Unsupported Media Type`: Unrecognized Content-Type header.
- `503 Service Unavailable`: ASR model not yet operational.

---

### POST /synthesize *(stub – 503)*

Synthesize speech from text in a Sierra Leonean language.

**Request**
```json
{
  "text": "TEST_FIXTURE_TEXT_001",
  "language": "krio"
}
```

**Errors**
- `422 Unprocessable Entity`: Language not supported or `"english"`.
- `503 Service Unavailable`: TTS model not yet operational.

---

## Running the API

```bash
# Start via CLI
slai serve

# Or with custom host/port
slai serve --host 127.0.0.1 --port 8000

# With auto-reload for development
slai serve --reload
```

---

## HTTP Status Codes

| Code | Meaning | Context |
|---|---|---|
| 200 | OK | Successful response |
| 400 | Bad Request | Empty or malformed audio file payload |
| 413 | Payload Too Large | Upload size exceeds `SLAI_MAX_AUDIO_UPLOAD_BYTES` |
| 415 | Unsupported Media Type | Non-audio MIME type sent to `/transcribe` |
| 422 | Unprocessable Entity | Unsupported language code or invalid translation pair |
| 503 | Service Unavailable | Model not yet operational (all model endpoints return this until models are loaded) |
