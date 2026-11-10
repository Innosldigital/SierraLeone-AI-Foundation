# Sierra Leone AI Foundation

> **Building open AI for Sierra Leonean languages.**

The Sierra Leone AI Foundation is an independent research and development
platform for language AI covering Sierra Leone's languages: **Krio** (primary),
Mende, Temne, Limba, Loko, Kono, Kissi and others.

The first working model — **CAFP Krio AI** — will be a Krio language model
built by adapting an open-weight multilingual base model through continued
pre-training and instruction fine-tuning.

---

## ⚠️ Current Status

| Component | Status |
|---|---|
| Project structure | ✅ Created |
| Dataset schemas | ✅ Defined |
| Preprocessing pipeline | ✅ Scaffolded |
| Evaluation framework | ✅ Scaffolded |
| API (stubs) | ✅ Created |
| Krio training data | 🔴 **Collection required** |
| Base model selection | 🟡 Research required |
| Continued pre-training | 🔴 Awaiting data + hardware |
| Speech recognition | 🔴 Awaiting audio data |
| Text-to-speech | 🔴 Awaiting audio data |

> No production model is currently operational.

---

## Languages

| Language | ISO 639-3 | Status |
|---|---|---|
| Krio | kri | 🟢 Active – primary language |
| Mende | men | 🔵 Planned |
| Temne | tem | 🔵 Planned |
| Limba | lia | 🔵 Planned |
| Loko | lok | 🔵 Planned |
| Kono | kno | 🔵 Planned |
| Kissi | kss | 🔵 Planned |

---

## Project Structure

```
SierraLeone-AI-Foundation/
├── datasets/           # Language datasets (raw, cleaned, validated)
├── models/             # Model weights and configs (not committed to git)
├── training/           # Training pipeline configurations
├── benchmarks/         # Held-out evaluation sets (never trained on)
├── src/
│   └── sierraleone_ai/ # Python package
│       ├── api/        # FastAPI application
│       ├── datasets/   # Dataset loading and schemas
│       ├── evaluation/ # Evaluation metrics and human eval forms
│       ├── models/     # Model registry and stubs
│       ├── preprocessing/ # Text normalisation
│       └── training/   # Training pipeline stubs
├── tests/              # Unit tests
├── documentation/      # Full documentation
├── research/           # Research notes and model comparisons
└── api/                # API deployment configs
```

---

## Quick Start

```bash
# 1. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/macOS

# 2. Install the package in development mode
pip install -e ".[dev]"

# 3. Copy and configure environment
copy .env.example .env

# 4. Run tests
pytest

# 5. Check project status
slai status

# 6. Start the API (stubs only – no model is live)
uvicorn sierraleone_ai.api.app:app --reload
```

---

## Important Restrictions

1. **Do not modify the CAFP Farmer App.** This is a separate project.
2. **Do not expose API keys.** Never commit `.env`.
3. **Do not invent datasets.** All training data must be real and attributed.
4. **Do not claim models are trained** when they are not.
5. **Do not start GPU training** without setting `SLAI_ALLOW_TRAINING=true` and obtaining approval.
6. **Do not confuse Krio with Nigerian Pidgin.** Sierra Leonean Krio is a distinct language.

---

## Documentation

See the [`documentation/`](documentation/) folder for:

- [System Architecture](documentation/SYSTEM_ARCHITECTURE.md)
- [Data Collection Guide](documentation/DATA_COLLECTION_GUIDE.md)
- [Dataset Format](documentation/DATASET_FORMAT.md)
- [Model Training Guide](documentation/MODEL_TRAINING_GUIDE.md)
- [Model Evaluation Guide](documentation/MODEL_EVALUATION_GUIDE.md)
- [API Documentation](documentation/API_DOCUMENTATION.md)
- [Development Roadmap](documentation/DEVELOPMENT_ROADMAP.md)
- [Dependencies and Licences](documentation/DEPENDENCIES_AND_LICENCES.md)

---

## Licence

Apache 2.0 — see [LICENCE](LICENCE).

Data collected from contributors is managed under separate consent agreements.
