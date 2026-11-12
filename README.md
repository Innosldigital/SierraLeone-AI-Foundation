# Sierra Leone AI Foundation

> **Building open AI for Sierra Leonean languages.**

The Sierra Leone AI Foundation is an independent research and development
platform for language AI covering Sierra Leone's languages: **Krio** (primary),
Mende, Temne, Limba, Loko, Kono, Kissi and others.

The first working model — **CAFP Krio AI** — follows an experiment ladder:
zero/few-shot candidate baseline evaluation against locked benchmarks, followed by
supervised fine-tuning (SFT) with LoRA adapters on native-reviewed data, and optional
continued pretraining (CPT) only if justified by benchmark gap analysis.

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
├── models/             # Model candidate registry and decisions (weights excluded from git)
├── training/           # Training pipeline configurations and readiness thresholds
├── benchmarks/         # Held-out evaluation sets (never trained on)
├── src/
│   └── sierraleone_ai/ # Python package
│       ├── api/        # FastAPI application (stubs until models are live)
│       ├── datasets/   # Dataset loading, schemas, eligibility, and manifests
│       ├── evaluation/ # Evaluation metrics, reporting, and native speaker forms
│       ├── models/     # Model candidate registry and tokeniser audit
│       ├── preprocessing/ # Text normalisation and speaker-aware split manager
│       ├── serving/    # In-memory model serving state (drives /health/ready)
│       ├── speech/     # ASR and TTS audio quality infrastructure
│       └── training/   # Training pipeline & safety guard
├── tests/              # Unit and integration test suite
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
pip install -e ".[dev,eval]"

# 3. Copy and configure environment
copy .env.example .env      # Windows
# cp .env.example .env      # Linux/macOS

# 4. Run tests
pytest

# 5. Check project status
slai status

# 6. Start the API (stubs only – returns 503 until models are operational)
slai serve
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
