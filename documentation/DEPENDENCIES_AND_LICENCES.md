# Dependencies and Licences

> [!WARNING]
> This file is maintained manually. Run `pip-licenses --format=markdown` after any
> dependency change to verify entries. Do not fill licence fields from memory.

In accordance with **CLAUDE.md Absolute Rule 5 (document every dependency and its licence)**, this file catalogues all core, optional, and development dependencies used by the **Sierra Leone AI Foundation** platform.

All dependencies use permissive or standard open-source licences compatible with the project's **Apache-2.0** licence.

---

## 1. Core Runtime Dependencies

| Package | Version Constraint | Licence | Purpose |
|---|---|---|---|
| `pydantic` | `>=2.5` | MIT | Dataset schema validation, provenance enforcement, API request/response validation |
| `pydantic-settings` | `>=2.1` | MIT | Environment variable and `.env` configuration management |
| `ftfy` | `>=6.1` | Apache-2.0 | Unicode mojibake repair and text normalisation |
| `fastapi` | `>=0.109` | MIT | Modular REST API framework (`/generate`, `/translate`, `/transcribe`, `/synthesize`, `/languages`, `/health`) |
| `uvicorn` | `>=0.25` | BSD-3-Clause | ASGI web server for FastAPI |
| `python-multipart` | `>=0.0.9` | MIT | Form-data and audio file upload parsing for `/transcribe` |
| `rich` | `>=13.7` | MIT | Terminal formatting and CLI status tables |
| `loguru` | `>=0.7` | MIT | Structured logging |
| `typer` | `>=0.9` | MIT | `slai` command-line interface |
| `python-dotenv` | `>=1.0` | BSD-3-Clause | `.env` file loading (ensuring secrets are never hardcoded) |

---

## 2. Optional Data & Analytics Dependencies (`pip install -e ".[data]"`)

| Package | Version Constraint | Licence | Purpose |
|---|---|---|---|
| `datasets` | `>=2.16` | Apache-2.0 | HuggingFace Datasets integration for large-scale streaming and arrow conversion |
| `pandas` | `>=2.1` | BSD-3-Clause | Tabular dataset statistics and evaluator sheet export |
| `pyarrow` | `>=14.0` | Apache-2.0 | Columnar Parquet storage for datasets exceeding 100k records |

---

## 3. Optional Model Training Dependencies (`pip install -e ".[training]"`)

> **Note:** Not installed by default. Install only on approved GPU training hosts after `SLAI_ALLOW_TRAINING=true` review.

| Package | Version Constraint | Licence | Purpose |
|---|---|---|---|
| `torch` | `>=2.1` | BSD-3-Clause | Deep learning tensor computation and autograd |
| `transformers` | `>=4.37` | Apache-2.0 | Base model loading, tokenisation analysis, and checkpoint management |
| `peft` | `>=0.8` | Apache-2.0 | LoRA / QLoRA parameter-efficient fine-tuning adapters |
| `bitsandbytes` | `>=0.42` | MIT | 4-bit (NF4) and 8-bit quantization for QLoRA |
| `accelerate` | `>=0.26` | Apache-2.0 | Distributed and mixed-precision training orchestration |
| `trl` | `>=0.7` | Apache-2.0 | Supervised Fine-Tuning (`SFTTrainer`) for conversational and QA data |
| `evaluate` | `>=0.4` | Apache-2.0 | Metric orchestration |
| `sacrebleu` | `>=2.3` | Apache-2.0 | Reproducible BLEU and chrF++ calculation |

---

## 4. Optional Speech (ASR / TTS) Dependencies (`pip install -e ".[audio]"`)

| Package | Version Constraint | Licence | Purpose |
|---|---|---|---|
| `torchaudio` | `>=2.1` | BSD-2-Clause | Audio resampling and spectrogram extraction |
| `librosa` | `>=0.10` | ISC | Audio duration, sample-rate, and SNR analysis |
| `soundfile` | `>=0.12` | BSD-3-Clause | Lossless WAV/FLAC audio I/O |
| `openai-whisper` | `>=20231117` | MIT | Candidate ASR base architecture for Krio speech recognition |

---

## 5. Development & Quality Assurance Dependencies (`pip install -e ".[dev]"`)

| Package | Version Constraint | Licence | Purpose |
|---|---|---|---|
| `pytest` | `>=7.4` | MIT | Unit and integration test suite runner |
| `pytest-cov` | `>=4.1` | MIT | Test coverage measurement |
| `ruff` | `>=0.1` | MIT | Fast Python linter and code formatter |
| `mypy` | `>=1.8` | MIT | Static type checking |
| `httpx` | `>=0.26` | BSD-3-Clause | HTTP client used as FastAPI `TestClient` backend in tests |

