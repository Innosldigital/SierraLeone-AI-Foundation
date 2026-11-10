# Native Speaker Review Queue

This file lists every Krio or other Sierra Leonean language sample that
currently appears in code, tests, or documentation and has NOT been
verified by a native Sierra Leonean Krio speaker.

**Rule:** No new Krio sentences are added by AI agents. All samples here
must be replaced with native-speaker-verified content or neutral English
placeholders before they can be used for training.

---

## Status Legend

| Symbol | Meaning |
|--------|---------|
| ⚠️ | Unverified — present in repo; must not be used for training |
| ✅ | Verified by native speaker (name + date required) |
| 🔄 | Replaced with neutral placeholder (action complete) |

---

## Queue

| # | Sample text | File(s) | Line(s) | Issue | Status |
|---|---|---|---|---|---|
| 1 | `"Mi de go maket tumara mɔning fɔ bai som rais en fish."` | `research/model_comparison.md`, `documentation/MODEL_TRAINING_GUIDE.md`, `preprocessing/krio_normalizer.py` (docstring), `DATASET_FORMAT.md`, `DATA_COLLECTION_GUIDE.md` | Various | Invented by AI; `Mi de go` may be Nigerian-Pidgin-style (SL Krio typically `A de go`); spelling inconsistent (`mɔning` vs `mɔnin`) | ⚠️ Unverified |
| 2 | `"Di rein don stat fɔl sotei di wata dɔn kɔvɔ di rod."` | `documentation/DATASET_FORMAT.md` | Example record | Invented by AI; not verified | ⚠️ Unverified |
| 3 | `"Gud mɔnin, aw di fam wok de go?"` | `documentation/DATASET_FORMAT.md` | Translation example | Invented by AI; not verified | ⚠️ Unverified |
| 4 | `"Wetin na di bɛs taim fɔ plant kasada?"` | `documentation/DATASET_FORMAT.md`, `documentation/DATA_COLLECTION_GUIDE.md`, `datasets/krio/conversations/example_agricultural_conversations_train.jsonl` | Conversation example | Invented by AI; not verified | ⚠️ Unverified |
| 5 | `"Di bɛs taim na wen di rein si don bigin, bifo di drai sizon."` | `documentation/DATASET_FORMAT.md`, `datasets/krio/conversations/example_agricultural_conversations_train.jsonl` | Conversation example | Invented by AI; agronomically unverified; not linguistically verified | ⚠️ Unverified |
| 6 | `"Kusheh! Aw di bodi?"` | `datasets/krio/cleaned/example_krio_text_train.jsonl`, `benchmarks/krio/held_out_eval_schema.jsonl`, `tests/test_schema.py`, `tests/test_normalizer.py`, `tests/test_speech_and_training.py` | Multiple | Invented by AI; common greeting — likely accurate but unverified | ⚠️ Unverified |
| 7 | `"Kusheh, aw di bodi? A wel tenki."` | `preprocessing/krio_normalizer.py` | Comment in `NON_KRIO_CREOLE_MARKERS` example | Invented by AI; not verified | ⚠️ Unverified |
| 8 | `"Gud mɔning"` | `tests/test_schema.py` | Translation test fixture | Invented by AI; not verified | ⚠️ Unverified |
| 9 | `"Kusheh"` | `tests/test_schema.py`, `tests/test_speech_and_training.py` | Test fixture | Invented by AI; not verified | ⚠️ Unverified |
| 10 | `"Mi de go maket."` | `tests/test_schema.py`, `tests/test_normalizer.py`, `tests/test_api.py` | Test fixtures | Invented by AI | ⚠️ Unverified |
| 11 | `"Wetin gud taim fɔ plant kasada?"` | `src/sierraleone_ai/api/app.py` (removed), `datasets/krio/conversations/` | — | Invented by AI | ⚠️ Unverified |

---

## Action Plan

**Tests and code (immediate):** Replace all Krio fixtures in `tests/` with
neutral English or opaque placeholders (`"TEST_FIXTURE_TEXT_001"`, `"FIXTURE_EN_001"`).
See audit finding SLAI-005f. AI agents must not write new Krio.

**Documentation:** Mark all remaining Krio examples with:
```
<!-- UNVERIFIED: needs native Sierra Leonean Krio speaker review -->
```

**Data pipeline:** The example JSONL files in `datasets/krio/` and
`benchmarks/krio/` are marked `SCHEMA_TEMPLATE_NOT_TRAINING_DATA` — they
will never be loaded as training data (enforced by the loader's eligibility gate).
They still need native-speaker review before being promoted to real data files.

---

## How to Mark a Sample as Verified

1. A native Sierra Leonean Krio speaker (born/raised in Sierra Leone, not AI) reviews the sample.
2. They sign a `ReviewerSignoff` record in the dataset metadata.
3. Update this table: change status to ✅, add `Verified by: [anonymised ID] on [date]`.
4. Remove the `<!-- UNVERIFIED -->` marker from documentation.
