# Dataset Format Specification

Version: 0.1.0

All datasets use **JSONL** (JSON Lines) format: one JSON object per line,
UTF-8 encoded, no trailing commas.

Large datasets (>100k records) may be stored as **Parquet** using the same schema.

---

## Common Metadata Fields

Every record MUST include a `metadata` object:

| Field | Type | Required | Description |
|---|---|---|---|
| `record_id` | string (UUID) | auto | Unique record identifier |
| `language` | enum | ✅ | `krio`, `mende`, `temne`, `english`, etc. |
| `source` | string | ✅ | Origin of the data (non-empty) |
| `collection_date` | date (YYYY-MM-DD) | ✅ | When the data was collected |
| `licence` | enum | ✅ | `CC-BY-4.0`, `CC0-1.0`, `Apache-2.0`, etc. |
| `validation_status` | enum | ✅ | See Validation Status below |
| `dataset_version` | string (semver) | ✅ | e.g. `0.1.0` |
| `split` | enum | ✅ | `train`, `validation`, `test`, `held_out` |
| `contributor_consent` | boolean | ✅ | `true` if consent obtained |
| `notes` | string | optional | Free-text notes |

### Validation Status Values

| Value | Meaning |
|---|---|
| `unvalidated` | Raw, not yet reviewed |
| `machine_generated` | Produced by MT/TTS – NOT ground truth |
| `partially_validated` | Reviewed by one native speaker |
| `validated` | Reviewed by ≥2 independent native speakers |
| `rejected` | Contains errors – do not use for training |

---

## Krio Text Record

```json
{
  "metadata": {
    "language": "krio",
    "source": "cafp_text_collection_2024_batch_01",
    "collection_date": "2024-03-10",
    "licence": "CC-BY-4.0",
    "validation_status": "validated",
    "dataset_version": "0.1.0",
    "split": "train",
    "contributor_consent": true
  },
  "text": "Di rein don stat fɔl.",
  "script": "Latin",
  "domain": "agriculture"
}
```

---

## Translation Record

```json
{
  "metadata": {
    "language": "krio",
    "source": "cafp_translation_2024",
    "collection_date": "2024-04-01",
    "licence": "CC-BY-4.0",
    "validation_status": "validated",
    "dataset_version": "0.1.0",
    "split": "train",
    "contributor_consent": true
  },
  "source_language": "english",
  "target_language": "krio",
  "source_text": "The rain has started falling.",
  "target_text": "Di rein don stat fɔl.",
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
    "language": "krio",
    "source": "cafp_conversation_collection_2024",
    "collection_date": "2024-05-01",
    "licence": "CC-BY-4.0",
    "validation_status": "validated",
    "dataset_version": "0.1.0",
    "split": "train",
    "contributor_consent": true
  },
  "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
  "domain": "agriculture",
  "turns": [
    {
      "speaker": "user",
      "language": "krio",
      "text": "Wetin gud taim fɔ plant kasada?",
      "turn_index": 0
    },
    {
      "speaker": "assistant",
      "language": "krio",
      "text": "Di bɛs taim na wen di rein si don bigin, bifo di drai sizon.",
      "turn_index": 1
    }
  ]
}
```

---

## Question-Answer Record

```json
{
  "metadata": {
    "language": "krio",
    "source": "cafp_qa_2024",
    "collection_date": "2024-06-01",
    "licence": "CC-BY-4.0",
    "validation_status": "validated",
    "dataset_version": "0.1.0",
    "split": "train",
    "contributor_consent": true
  },
  "question": "Aw fɔ plant tomato?",
  "answer": "Fɔs, pri da sɔil mek i bin soft. Den put di sid ...",
  "question_language": "krio",
  "answer_language": "krio",
  "domain": "agriculture",
  "is_factual": true
}
```

---

## Audio Record

```json
{
  "metadata": {
    "language": "krio",
    "source": "cafp_audio_collection_2024",
    "collection_date": "2024-07-01",
    "licence": "CC-BY-4.0",
    "validation_status": "unvalidated",
    "dataset_version": "0.1.0",
    "split": "train",
    "contributor_consent": true
  },
  "audio_file_path": "krio/audio/spk_001/rec_001.wav",
  "transcript": "Mi de go maket tumara.",
  "duration_seconds": 2.4,
  "sample_rate_hz": 16000,
  "speaker_id": "spk_001",
  "gender": "female",
  "age_group": "adult",
  "recording_environment": "home",
  "snr_db": null
}
```

---

## File Naming Convention

```
datasets/{language}/{stage}/{type}_{split}.jsonl
```

Examples:
- `datasets/krio/cleaned/krio_text_train.jsonl`
- `datasets/krio/cleaned/krio_text_validation.jsonl`
- `datasets/krio/translations/translations_english_krio_train.jsonl`
- `datasets/krio/conversations/conversations_train.jsonl`
