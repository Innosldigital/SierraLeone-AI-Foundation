# Dataset Format Specification

Version: 0.2.0 (Schema v2)

All datasets use **JSONL** (JSON Lines) format: one JSON object per line,
UTF-8 encoded, no trailing commas.

Large datasets (>100k records) may be stored as **Parquet** using the same schema.

---

## Common Metadata Fields (Schema v2)

Every record MUST include a `metadata` object complying with `DatasetMetadata` (schema version 2):

| Field | Type | Required | Description |
|---|---|---|---|
| `schema_version` | integer | ✅ | Must be `2` |
| `record_id` | string (UUID) | ✅ | Unique record identifier (UUID v4 format) |
| `language` | enum | ✅ | `krio`, `mende`, `temne`, `english`, etc. |
| `source` | string | ✅ | Origin of data (must be approved in `source_register.csv`) |
| `collection_date` | date (YYYY-MM-DD) | ✅ | When the data was collected |
| `licence` | string | ✅ | Display licence identifier |
| `licence_id` | enum | ✅ | Canonical licence (e.g. `CC-BY-4.0`, `CC0-1.0`, `Apache-2.0`) |
| `validation_status` | enum | ✅ | See Validation Status below |
| `dataset_version` | string (semver) | ✅ | e.g. `0.2.0` |
| `split` | enum | ✅ | `train`, `validation`, `test`, `held_out` |
| `contributor_id` | string | ✅ | Pseudonymised contributor ID |
| `contributor_consent` | boolean | ✅ | `true` if consent obtained |
| `consent_scope` | list[enum] | ✅ | e.g. `["text_training"]`, `["speech_recognition"]` |
| `access_class` | enum | ✅ | `public`, `restricted`, `internal_only` |
| `training_permitted` | boolean | ✅ | Explicit training clearance (default `false`) |
| `redistribution_permitted` | boolean | ✅ | Explicit redistribution clearance (default `false`) |
| `reviews` | list[ReviewerSignoff] | optional | Review sign-offs (VALIDATED requires ≥2 independent native approvals) |

### Validation Status Values

| Value | Meaning |
|---|---|
| `unvalidated` | Raw, not yet reviewed |
| `machine_generated` | Produced by MT/TTS – NOT ground truth |
| `partially_validated` | Reviewed by 1 native speaker |
| `validated` | Reviewed by ≥2 independent native speakers with valid consent |
| `rejected` | Contains errors or withdrawn – do not use for training |

---

## Krio Text Record

```json
{
  "metadata": {
    "schema_version": 2,
    "record_id": "01234567-89ab-4cde-8f01-23456789abcd",
    "language": "krio",
    "source": "test_fixture_approved_source",
    "collection_date": "2026-01-15",
    "licence": "CC-BY-4.0",
    "licence_id": "CC-BY-4.0",
    "validation_status": "validated",
    "dataset_version": "0.2.0",
    "split": "train",
    "contributor_id": "contrib_001",
    "contributor_consent": true,
    "consent_scope": [
      "text_training"
    ],
    "access_class": "public",
    "training_permitted": true,
    "redistribution_permitted": true,
    "reviews": [
      {
        "reviewer_id": "rev_sl_001",
        "is_native_speaker_of_record_language": true,
        "reviewed_on": "2026-01-16",
        "verdict": "approve"
      },
      {
        "reviewer_id": "rev_sl_002",
        "is_native_speaker_of_record_language": true,
        "reviewed_on": "2026-01-17",
        "verdict": "approve"
      }
    ]
  },
  "text": "TEST_FIXTURE_KRIO_TEXT_001",
  "script": "Latin",
  "domain": "agriculture"
}
```

---

## Translation Record

```json
{
  "metadata": {
    "schema_version": 2,
    "record_id": "01234567-89ab-4cde-8f01-23456789abcd",
    "language": "krio",
    "source": "test_fixture_approved_source",
    "collection_date": "2026-01-15",
    "licence": "CC-BY-4.0",
    "licence_id": "CC-BY-4.0",
    "validation_status": "validated",
    "dataset_version": "0.2.0",
    "split": "train",
    "contributor_id": "contrib_001",
    "contributor_consent": true,
    "consent_scope": [
      "text_training"
    ],
    "access_class": "public",
    "training_permitted": true,
    "redistribution_permitted": true,
    "reviews": [
      {
        "reviewer_id": "rev_sl_001",
        "is_native_speaker_of_record_language": true,
        "reviewed_on": "2026-01-16",
        "verdict": "approve"
      },
      {
        "reviewer_id": "rev_sl_002",
        "is_native_speaker_of_record_language": true,
        "reviewed_on": "2026-01-17",
        "verdict": "approve"
      }
    ]
  },
  "source_language": "english",
  "target_language": "krio",
  "source_text": "TEST_FIXTURE_ENGLISH_SOURCE_001",
  "target_text": "TEST_FIXTURE_KRIO_TARGET_001",
  "translation_method": "human",
  "back_translation_verified": true
}
```

Allowed `translation_method` values: `human`, `machine_then_human_post_edit`, `machine_only`.

---

## Conversation Record

```json
{
  "metadata": {
    "schema_version": 2,
    "record_id": "01234567-89ab-4cde-8f01-23456789abcd",
    "language": "krio",
    "source": "test_fixture_approved_source",
    "collection_date": "2026-01-15",
    "licence": "CC-BY-4.0",
    "licence_id": "CC-BY-4.0",
    "validation_status": "validated",
    "dataset_version": "0.2.0",
    "split": "train",
    "contributor_id": "contrib_001",
    "contributor_consent": true,
    "consent_scope": [
      "text_training"
    ],
    "access_class": "public",
    "training_permitted": true,
    "redistribution_permitted": true,
    "reviews": [
      {
        "reviewer_id": "rev_sl_001",
        "is_native_speaker_of_record_language": true,
        "reviewed_on": "2026-01-16",
        "verdict": "approve"
      },
      {
        "reviewer_id": "rev_sl_002",
        "is_native_speaker_of_record_language": true,
        "reviewed_on": "2026-01-17",
        "verdict": "approve"
      }
    ]
  },
  "conversation_id": "00000000-0000-4000-8000-000000000002",
  "domain": "general",
  "turns": [
    {
      "speaker": "user",
      "language": "krio",
      "text": "TEST_FIXTURE_USER_PROMPT_001",
      "turn_index": 0
    },
    {
      "speaker": "assistant",
      "language": "krio",
      "text": "TEST_FIXTURE_ASSISTANT_RESPONSE_001",
      "turn_index": 1
    }
  ],
  "num_turns": 2
}
```

---

## Audio Record

```json
{
  "metadata": {
    "schema_version": 2,
    "record_id": "01234567-89ab-4cde-8f01-23456789abcd",
    "language": "krio",
    "source": "test_fixture_approved_source",
    "collection_date": "2026-01-15",
    "licence": "CC-BY-4.0",
    "licence_id": "CC-BY-4.0",
    "validation_status": "validated",
    "dataset_version": "0.2.0",
    "split": "train",
    "contributor_id": "contrib_001",
    "contributor_consent": true,
    "consent_scope": [
      "speech_recognition",
      "voice_synthesis"
    ],
    "access_class": "public",
    "training_permitted": true,
    "redistribution_permitted": true,
    "reviews": [
      {
        "reviewer_id": "rev_sl_001",
        "is_native_speaker_of_record_language": true,
        "reviewed_on": "2026-01-16",
        "verdict": "approve"
      },
      {
        "reviewer_id": "rev_sl_002",
        "is_native_speaker_of_record_language": true,
        "reviewed_on": "2026-01-17",
        "verdict": "approve"
      }
    ]
  },
  "audio_file_path": "audio/spk_01/recording_001.wav",
  "transcript": "TEST_FIXTURE_TRANSCRIPT_001",
  "duration_seconds": 3.5,
  "sample_rate_hz": 24000,
  "speaker_id": "spk_01",
  "recording_environment": "studio",
  "snr_db": 32.0
}
```

---

## Benchmark Item Record

```json
{
  "metadata": {
    "schema_version": 2,
    "record_id": "00000000-0000-4000-8000-000000000003",
    "language": "krio",
    "source": "test_fixture_benchmark_collection",
    "collection_date": "2026-01-20",
    "licence": "CC-BY-4.0",
    "licence_id": "CC-BY-4.0",
    "validation_status": "validated",
    "dataset_version": "0.2.0",
    "split": "held_out",
    "contributor_id": "contrib_bench_01",
    "contributor_consent": true,
    "consent_scope": [
      "evaluation"
    ],
    "access_class": "public",
    "training_permitted": false,
    "redistribution_permitted": true,
    "reviews": [
      {
        "reviewer_id": "rev_sl_001",
        "is_native_speaker_of_record_language": true,
        "reviewed_on": "2026-01-21",
        "verdict": "approve"
      },
      {
        "reviewer_id": "rev_sl_002",
        "is_native_speaker_of_record_language": true,
        "reviewed_on": "2026-01-22",
        "verdict": "approve"
      }
    ]
  },
  "item_id": "bench_krio_001",
  "evaluation_category": "translation",
  "prompt": "TEST_FIXTURE_BENCHMARK_PROMPT_001",
  "references": [
    "TEST_FIXTURE_BENCHMARK_REF_001"
  ],
  "reference_notes": "Standard Freetown dialect variant",
  "review_count": 2,
  "benchmark_version": "1.0.0"
}
```

---

## File Naming Convention

```
datasets/{language}/{stage}/{type}_{split}.jsonl
```

Examples:
- `datasets/krio/validated/krio_text_train.jsonl`
- `datasets/krio/validated/krio_text_validation.jsonl`
- `datasets/krio/translations/translations_english_krio_train.jsonl`
- `datasets/krio/conversations/conversations_train.jsonl`
- `benchmarks/krio/held_out_eval.jsonl`
