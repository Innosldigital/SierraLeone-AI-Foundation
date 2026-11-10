# CLAUDE.md — Sierra Leone AI Foundation

> This file provides project-specific context, conventions, and rules for AI
> coding assistants (Claude, Antigravity, Gemini, etc.) working in this repository.
> Read this file before making any changes to the codebase.

---

## Project Identity

**Sierra Leone AI Foundation** — an independent AI research and development
platform for Sierra Leonean languages.

- **Primary language:** Krio (ISO 639-3: `kri`) — Sierra Leonean Krio only
- **First model:** CAFP Krio AI (language model, not yet trained)
- **This project is NOT the CAFP Farmer App** — do not modify that app

---

## Absolute Rules

These rules override everything else. Never break them.

1. **Never fabricate training data.** Do not write example Krio sentences,
   translations, or QA pairs and pass them off as real data. Schema templates
   must be clearly marked `SCHEMA_TEMPLATE_NOT_TRAINING_DATA`.

2. **Never confuse Sierra Leonean Krio with Nigerian Pidgin, Liberian English,
   Cameroon Pidgin, or Ghanaian Pidgin.** They are distinct languages/creoles.
   Krio uses `duya` ("please"), `padi`/`bo` ("friend"), `ɛnti`/`noto so`
   (tag question) — not `abeg`, `chale`, or `shey`.

3. **Never claim a model is trained, operational, or fluent** unless it has been
   explicitly trained, evaluated by native speakers, and approved. All API
   model endpoints must return HTTP 503 until a real model is deployed.

4. **Never start GPU training automatically.** The `SLAI_ALLOW_TRAINING`
   environment variable must be explicitly set to `true` by a human before
   any training call proceeds. The `require_training_approval()` guard in
   `src/sierraleone_ai/training/guard.py` enforces this.

5. **Never expose API keys or secrets.** Use `.env` (not committed) and
   `.env.example` (committed, with empty values). Never hardcode keys.

6. **Never train on `machine_generated` data without clear labelling.**
   The `ValidationStatus.MACHINE_GENERATED` enum value must be preserved.
   Machine-generated translations must never have `validation_status = "validated"`.

7. **Never use or reference the `machine_only` translation method as
   validated ground truth.**

8. **Never modify the CAFP Farmer App** from within this repository.

---

## Repository Layout

```
SierraLeone-AI-Foundation/
├── src/sierraleone_ai/        ← Python package (install with pip install -e ".[dev]")
│   ├── __init__.py            ← SUPPORTED_LANGUAGES, __version__
│   ├── config.py              ← pydantic-settings config (env prefix: SLAI_)
│   ├── cli.py                 ← `slai` CLI (typer)
│   ├── api/                   ← FastAPI app (stubs only until models are live)
│   ├── datasets/              ← Pydantic schemas + JSONL loaders
│   ├── evaluation/            ← BLEU/WER/CER metrics + native speaker forms
│   ├── models/                ← Candidate base model registry (no weights)
│   ├── preprocessing/         ← Krio normaliser + speaker-aware split manager
│   ├── speech/                ← ASR + TTS infrastructure stubs
│   └── training/              ← Training guard + LoRA/CPT/SFT pipeline stubs
├── datasets/                  ← Language datasets (JSONL / Parquet)
│   └── krio/                  ← raw/ cleaned/ validated/ translations/ conversations/ audio/
├── benchmarks/                ← Held-out evaluation sets (NEVER train on these)
├── models/                    ← Model weights (excluded from git via .gitignore)
├── training/                  ← Training configs (JSON)
├── documentation/             ← Full project documentation
├── research/                  ← Base model comparison notes
├── tests/                     ← pytest suite (35 tests, all passing)
├── api/                       ← FastAPI entrypoint for deployment
├── pyproject.toml             ← Project metadata and dependencies
├── .env.example               ← Environment variable template
└── CLAUDE.md                  ← This file
```

---

## Python Environment

```bash
# Recommended Python version: >= 3.10 (3.10, 3.12, 3.13 tested in CI)

# Create and activate virtual environment:
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate
# On Windows PowerShell:
.venv\Scripts\Activate.ps1

# Install in dev mode:
pip install -e ".[dev,eval]"

# Run tests:
pytest -v

# Install optional HuggingFace data tools:
pip install -e ".[data]"

# Install training deps (only on approved GPU host with SLAI_ALLOW_TRAINING=true):
pip install -e ".[training]"
```

---

## Key Source Files

| File | Purpose |
|------|---------|
| [`src/sierraleone_ai/datasets/schema.py`](src/sierraleone_ai/datasets/schema.py) | All Pydantic dataset record types — edit here when adding new dataset schemas |
| [`src/sierraleone_ai/preprocessing/krio_normalizer.py`](src/sierraleone_ai/preprocessing/krio_normalizer.py) | Krio text normalisation + cross-creole contamination detection |
| [`src/sierraleone_ai/evaluation/native_speaker.py`](src/sierraleone_ai/evaluation/native_speaker.py) | Human evaluation forms for native Krio speaker review |
| [`src/sierraleone_ai/models/registry.py`](src/sierraleone_ai/models/registry.py) | Candidate base model metadata (no weights downloaded here) |
| [`src/sierraleone_ai/training/guard.py`](src/sierraleone_ai/training/guard.py) | Safety gate: `require_training_approval()` — call at top of every training entry point |
| [`src/sierraleone_ai/training/pipeline.py`](src/sierraleone_ai/training/pipeline.py) | CPT + SFT + LoRA pipeline configurations and guarded entry points |
| [`src/sierraleone_ai/speech/asr.py`](src/sierraleone_ai/speech/asr.py) | ASR infrastructure + speaker leakage audit |
| [`src/sierraleone_ai/speech/tts.py`](src/sierraleone_ai/speech/tts.py) | TTS training quality checks |
| [`src/sierraleone_ai/api/app.py`](src/sierraleone_ai/api/app.py) | FastAPI app — all model endpoints return 503 until operational |
| [`src/sierraleone_ai/config.py`](src/sierraleone_ai/config.py) | Settings — all env vars prefixed `SLAI_` |

---

## Dataset Schema Conventions

- Every dataset record **must** have a `metadata: DatasetMetadata` object.
- Required metadata fields: `language`, `source`, `collection_date`, `licence`,
  `validation_status`, `dataset_version`, `split`, `contributor_consent`.
- `source` must be a non-empty descriptive string (not `"unknown"`).
- `contributor_consent` must be `true` for any real collected data.
- `validation_status` values in priority order:
  `unvalidated` → `partially_validated` → `validated` (requires 2 independent native speakers)
- `machine_generated` data must never be promoted to `validated`.
- Speaker IDs in audio records **must** be assigned before splitting to prevent leakage.

**Dataset file naming:**
```
datasets/{language}/{stage}/{type}_{split}.jsonl
```
Example: `datasets/krio/cleaned/krio_text_train.jsonl`

---

## Adding a New Language

1. Add the language to `SUPPORTED_LANGUAGES` in `src/sierraleone_ai/__init__.py`
2. Create the directory structure: `datasets/{language_code}/{raw,cleaned,validated,translations,conversations,audio}/`
3. Add a `.gitkeep` with a comment to the top-level language directory
4. Add a benchmark directory: `benchmarks/{language_code}/`
5. No changes to the API, schemas, or core pipeline are needed — they are already multilingual.

---

## Adding New API Endpoints

1. Add the request/response Pydantic models to `src/sierraleone_ai/api/models.py`
2. Add the endpoint to `src/sierraleone_ai/api/app.py`
3. If the model is not yet operational, raise `_MODEL_NOT_READY` (HTTP 503)
4. Add a `test_*_returns_503` test to `tests/test_api.py`
5. Document the endpoint in `documentation/API_DOCUMENTATION.md`

---

## Testing

```bash
# Run all unit and integration tests:
pytest -v

# Run a specific test file:
pytest tests/test_schema.py -v

# Run with test coverage:
pytest -v --cov=sierraleone_ai --cov-report=term-missing
```

**Test files (110+ tests, all passing):**
- `tests/test_schema.py` — Pydantic schema validation, provenance rules, reviewer signoffs
- `tests/test_normalizer.py` — Krio normaliser, orthography profiling, creole contamination detection
- `tests/test_metrics.py` — BLEU (smoothed), CER, WER + native speaker eval forms
- `tests/test_api.py` — FastAPI endpoints (health, languages, 503 stubs, 422 language validation)
- `tests/test_speech_and_training.py` — ASR leakage audit, TTS quality gate, training guard
- `tests/test_loader.py` — JSONL loader eligibility gate, consent, template/split filters
- `tests/test_readiness.py` — Training guard dataset thresholds & readiness verification
- `tests/test_benchmarks.py` — Held-out integrity and text fingerprint contamination checks
- `tests/test_split_manager.py` — Deterministic largest-remainder splitting & speaker leakage prevention
- `tests/test_config.py` — Pydantic settings, secrets handling, path resolution
- `tests/test_registry.py` — Model registry candidate validation & research recommendations
- `tests/test_cli.py` — CLI command verification (`version`, `languages`, `status`, help)
- `tests/test_configs_json.py` — Validation of training pipeline JSON configs

---

## Documentation Files

| Document | Location |
|----------|----------|
| Project overview | [`README.md`](README.md) |
| System architecture | [`documentation/SYSTEM_ARCHITECTURE.md`](documentation/SYSTEM_ARCHITECTURE.md) |
| Data collection guide | [`documentation/DATA_COLLECTION_GUIDE.md`](documentation/DATA_COLLECTION_GUIDE.md) |
| Dataset format spec | [`documentation/DATASET_FORMAT.md`](documentation/DATASET_FORMAT.md) |
| Model training guide | [`documentation/MODEL_TRAINING_GUIDE.md`](documentation/MODEL_TRAINING_GUIDE.md) |
| Evaluation guide | [`documentation/MODEL_EVALUATION_GUIDE.md`](documentation/MODEL_EVALUATION_GUIDE.md) |
| API documentation | [`documentation/API_DOCUMENTATION.md`](documentation/API_DOCUMENTATION.md) |
| Development roadmap | [`documentation/DEVELOPMENT_ROADMAP.md`](documentation/DEVELOPMENT_ROADMAP.md) |
| Dependencies & licences | [`documentation/DEPENDENCIES_AND_LICENCES.md`](documentation/DEPENDENCIES_AND_LICENCES.md) |
| Base model comparison | [`research/model_comparison.md`](research/model_comparison.md) |
| Native review queue | [`docs/NATIVE_REVIEW_QUEUE.md`](docs/NATIVE_REVIEW_QUEUE.md) |

---

## Current Project Status (Phase 1 Complete + Full Audit Applied)

| Component | Status |
|-----------|--------|
| Project structure | ✅ Complete |
| Python package (`sierraleone_ai`) | ✅ Installed |
| Dataset schemas (all 8 types) | ✅ Enforced dual reviewer signoff + structured consent |
| Krio normaliser + creole contamination detection | ✅ Working (hard/soft markers, orthography profiling) |
| Speaker-aware split manager | ✅ Working (largest-remainder allocation, zero leakage) |
| ASR infrastructure stubs | ✅ Scaffolded |
| TTS infrastructure stubs | ✅ Scaffolded with strict studio quality gates |
| Training pipeline stubs (CPT + SFT + LoRA) | ✅ Scaffolded & guarded with dataset thresholds |
| Native speaker evaluation framework | ✅ Validated |
| FastAPI (6 endpoints, all correctly stubbed) | ✅ Running (informative 503s + 422 validations) |
| `slai` CLI | ✅ Installed & functional |
| Unit test suite | ✅ 110/110 passing |
| Full documentation suite (10 docs) | ✅ Written & audited |
| **Krio training data** | 🔴 Collection required |
| **Base model selection** | 🟡 Tokenisation audit required |
| **Continued pretraining** | 🔴 Awaiting data + hardware |
| **Speech recognition** | 🔴 Awaiting audio data |
| **Text-to-speech** | 🔴 Awaiting audio data |

---

## What to Do Next (Human Actions Required)

1. **Recruit native Sierra Leonean Krio speakers** for data collection and validation.
2. **Collect 50,000+ Krio text sentences** using the schema in `datasets/krio/cleaned/`.
3. **Run tokenisation analysis** on candidate models (see `research/model_comparison.md`).
4. **Select a base model** and document the decision before downloading weights.
5. **Never proceed to training** without completing steps 1–4.

