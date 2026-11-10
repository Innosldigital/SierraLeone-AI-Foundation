# Data Collection Guide

## Principles

1. **Obtain informed consent** from all contributors before recording or transcribing.
2. **Never fabricate data.** All text must come from real speakers or verified sources.
3. **Label machine-generated content** as `machine_generated` – never use it as ground truth.
4. **Attribute every record** with source, date and licence.
5. **Separate speakers** across train/validation/test to prevent leakage.

---

## Krio Data Priorities

### Priority 1 – Krio Text

> [!CAUTION]
> **BBC Pidgin is NOT Krio.** The BBC's West Africa service is BBC Pidgin (launched 2017),
> which is Nigerian Pidgin — a different creole language. Collecting BBC Pidgin text
> as Krio data would violate Absolute Rule 2 (do not confuse SL Krio with Nigerian Pidgin).
> Do NOT collect from BBC Pidgin / BBC Africa sources without expert verification.
>
> **Google Translate Krio (added May 2022)** outputs are machine-generated. They may only
> be collected as `machine_generated`-labelled data and must never appear in the validation
> or held-out benchmark sets.

Collect naturally occurring Sierra Leonean Krio text from:

- Native Krio speakers (written essays, messages, stories)
- Verified Krio publications and newspapers (written permission required — contact publisher)
- Umeå University Krio Corpus Project (contact corpus team for licence/sharing terms)
- Krio scriptural translations — e.g. Krio New Testament (confirm rights-holder before use)
- Sierra Leone radio/TV broadcasters (written permission required)
- Official Sierra Leone government Krio communications
- Community-contributed written text (with consent)

> [!NOTE]
> **Collaborate before collecting.** Two Sierra Leone Krio projects received LINGUA Africa
> funding in 2026: *Krio Voices* (Christex Foundation) — open speech + text corpus; and
> *SAM App* (Institute for Development) — Krio telemedicine. Contact them before building
> a parallel collection effort. A shared orthography standard, consent forms, and held-out
> benchmark will benefit all parties.

**Do NOT collect:**
- Nigerian Pidgin, Liberian English or Cameroon Pidgin
- Automatic translations without labelling them as `machine_generated`
- Copyrighted content without licence

### Priority 2 – English–Krio Parallel Data

Sentence-aligned pairs for translation training:

- Native speaker translations of English texts into Krio
- Bilingual documents (Krio + English)
- Manually created pairs reviewed by native speakers

Mark translation method as one of:
- `human` – translated entirely by a native speaker
- `machine_then_human_post_edit` – machine draft reviewed and corrected by native speaker
- `machine_only` – NOT suitable for supervised translation training

### Priority 3 – Krio Conversations

Multi-turn dialogues for instruction fine-tuning:

- Greetings and social exchanges
- Agricultural conversations (farming, crops, weather)
- Health conversations
- Market and commerce
- Education
- Government services

Each conversation must have a `domain` label and ≥2 turns.

### Priority 4 – Krio Audio

For ASR and TTS models:

- Recruit native Krio speakers with diverse backgrounds
- Collect in a range of recording environments (studio preferred)
- Obtain signed consent covering AI training use
- Record at minimum 16 kHz / 16-bit WAV
- Transcribe every recording using the standard audio schema
- Assign each speaker a persistent anonymised ID

---

## Record Format

See [DATASET_FORMAT.md](DATASET_FORMAT.md) for the full JSONL schema.

Minimum required fields:
```json
{
  "metadata": {
    "language": "krio",
    "source": "manual_collection_cafp_2024_batch_01",
    "collection_date": "2024-01-15",
    "licence": "CC-BY-4.0",
    "validation_status": "unvalidated",
    "dataset_version": "0.1.0",
    "split": "train",
    "contributor_consent": true
  },
  "text": "Mi de go maket tumara."
}
```

---

## Validation Workflow

```
Collect → unvalidated → Native speaker review 1
  → partially_validated → Native speaker review 2 (independent)
  → validated (ready for training)
```

Records marked `rejected` must never be used for training.

---

## Data Storage

- Raw files: `datasets/krio/raw/`
- After normalisation: `datasets/krio/cleaned/`
- After validation: `datasets/krio/validated/`
- Translations: `datasets/krio/translations/`
- Conversations: `datasets/krio/conversations/`
- Audio: `datasets/krio/audio/` (excluded from git)

Use JSONL format (one record per line) for text data.
Use Parquet for large datasets (>100k records).

---

## Minimum Dataset Requirements Before Training

| Dataset type | Minimum for initial fine-tuning |
|---|---|
| Krio text (cleaned) | 50,000 sentences |
| English–Krio translations | 10,000 pairs |
| Krio conversations | 1,000 dialogues |
| Held-out evaluation text | 1,000 sentences (validated) |
| Held-out translations | 500 pairs (validated) |

These are minimum estimates. More data is always better.
