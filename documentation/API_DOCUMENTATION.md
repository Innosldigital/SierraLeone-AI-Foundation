# API Documentation

Base URL: `http://localhost:8000` (development)

Interactive docs: `http://localhost:8000/docs`

> ⚠️ All model endpoints currently return HTTP 503. No model is operational.

---

## Endpoints

### GET /health

Returns the health status of the API.

**Response**
```json
{
  "status": "ok",
  "version": "0.1.0",
  "models_operational": [],
  "note": "Foundation phase – no models are operational yet."
}
```

---

### GET /languages

Lists all supported and planned languages.

**Response**
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

### POST /generate *(stub – 503)*

Generate text in a Sierra Leonean language.

**Request**
```json
{
  "prompt": "Kusheh, aw di bodi?",
  "language": "krio",
  "max_new_tokens": 256,
  "temperature": 0.7
}
```

**Response (when operational)**
```json
{
  "generated_text": "...",
  "language": "krio",
  "model_id": "cafp-krio-ai-v1",
  "tokens_generated": 128
}
```

---

### POST /translate *(stub – 503)*

Translate text between English and a Sierra Leonean language.

**Request**
```json
{
  "text": "Good morning, how are you?",
  "source_language": "english",
  "target_language": "krio"
}
```

---

### POST /transcribe *(stub – 503)*

Transcribe audio to text. Accepts multipart/form-data with an audio file.

---

### POST /synthesize *(stub – 503)*

Synthesize speech from text.

---

## Running the API

```bash
# Development
uvicorn sierraleone_ai.api.app:app --reload --host 0.0.0.0 --port 8000

# Production (when models are operational)
uvicorn sierraleone_ai.api.app:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## Error Codes

| Code | Meaning |
|---|---|
| 200 | Success |
| 400 | Bad request (invalid input) |
| 503 | Model not yet operational |
| 500 | Internal server error |
