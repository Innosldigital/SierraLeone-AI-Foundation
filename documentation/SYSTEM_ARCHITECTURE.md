# System Architecture

## Overview

The Sierra Leone AI Foundation is structured as a modular Python monorepo.
Each component is independently replaceable to support iterative improvements
as the project evolves from dataset collection through model training to production API.

```mermaid
graph TD
    A[Data Collection] --> B[Dataset Schemas & Validation]
    B --> C[Preprocessing Pipeline]
    C --> D[Training Data Store]
    D --> E[Continued Pre-training]
    E --> F[Instruction Fine-tuning / LoRA]
    F --> G[Model Evaluation]
    G --> H{Approved?}
    H -- Yes --> I[Model Registry]
    H -- No --> C
    I --> J[FastAPI Service]
    J --> K[/generate]
    J --> L[/translate]
    J --> M[/transcribe]
    J --> N[/synthesize]
```

## Components

### 1. `sierraleone_ai.datasets`
- **schema.py** – Pydantic models for all dataset record types
- **loader.py** – JSONL/Parquet loading with schema validation
- **stats.py** – Dataset audit and reporting

### 2. `sierraleone_ai.preprocessing`
- **krio_normalizer.py** – Unicode repair, whitespace, spelling normalisation
- **split_manager.py** – Speaker-aware train/validation/test splits

### 3. `sierraleone_ai.evaluation`
- **metrics.py** – BLEU, CER, WER (automated, supplementary only)
- **native_speaker.py** – Structured human evaluation forms
- **benchmarks.py** – Held-out benchmark management

### 4. `sierraleone_ai.models`
- **registry.py** – Candidate base model metadata (no weights downloaded)

### 5. `sierraleone_ai.training`
- **guard.py** – Safety gate preventing accidental training

### 6. `sierraleone_ai.api`
- **app.py** – FastAPI application with stub endpoints
- **models.py** – Request/response Pydantic models

## Data Flow

```
Raw text/audio → validation → normalisation → split assignment
     → JSONL/Parquet files → HuggingFace Dataset
     → (after approval) → training pipeline → model checkpoint
     → evaluation → (if approved) → API deployment
```

## Multilingual Design

The system is designed from the outset to support multiple languages.
Language codes are used consistently throughout:
- Dataset paths: `datasets/{language_code}/`
- Schema: `Language` enum
- API: `language` parameter on all endpoints
- Model registry: per-language model entries

Adding Mende, Temne or any other language requires:
1. Adding the language to `SUPPORTED_LANGUAGES` in `__init__.py`
2. Creating the dataset directory structure
3. Collecting and validating data
4. Training a language-specific or multilingual model

No code changes to the API or core pipeline are needed.

## Security and Privacy

- API keys are never committed to source control
- Contributor consent is tracked at the record level
- Speaker IDs are anonymised before storage
- Audio recordings are stored outside git (`.gitignore`)
- The `.env.example` template shows all required environment variables
