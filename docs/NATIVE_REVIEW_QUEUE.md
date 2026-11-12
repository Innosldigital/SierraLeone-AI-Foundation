# Native Speaker Review Queue

This file lists every Krio or other Sierra Leonean language sample that
previously appeared or currently appears in templates, code, tests, or documentation
and has NOT been verified by a native Sierra Leonean Krio speaker.

**Rule:** No new Krio sentences are added by AI agents. All test fixtures
use neutral placeholders (`TEST_FIXTURE_*`). Template files remain strictly
quarantined by `SCHEMA_TEMPLATE_NOT_TRAINING_DATA` and cannot be loaded for training.

---

## Status Legend

| Symbol | Meaning |
|--------|---------|
| ⚠️ | Unverified — present only in schema template; must not be used for training |
| ✅ | Verified by native speaker (name + date required) |
| 🔄 | Replaced with neutral placeholder (action complete) |

---

## Queue

| # | Sample text | File(s) | Line(s) | Issue | Status |
|---|---|---|---|---|---|
| 1 | `"Mi de go maket tumara mɔning fɔ bai som rais en fish."` | `research/model_comparison.md`, `documentation/MODEL_TRAINING_GUIDE.md` | Various | Unverified string in research notes; replaced in code/tests | 🔄 Replaced in code/tests; ⚠️ template only in research |
| 2 | `"Di rein don stat fɔl sotei di wata dɔn kɔvɔ di rod."` | `documentation/DATASET_FORMAT.md` | Example record | Replaced with `TEST_FIXTURE_KRIO_TEXT_001` | 🔄 Replaced |
| 3 | `"Gud mɔnin, aw di fam wok de go?"` | `documentation/DATASET_FORMAT.md` | Translation example | Replaced with `TEST_FIXTURE_KRIO_TARGET_001` | 🔄 Replaced |
| 4 | `"Wetin na di bɛs taim fɔ plant kasada?"` | `datasets/krio/conversations/example_agricultural_conversations_train.jsonl` | Conversation template | Invented by AI; strictly marked `SCHEMA_TEMPLATE_NOT_TRAINING_DATA` | ⚠️ Unverified (template only) |
| 5 | `"Di bɛs taim na wen di rein si don bigin, bifo di drai sizon."` | `datasets/krio/conversations/example_agricultural_conversations_train.jsonl` | Conversation template | Invented by AI; strictly marked `SCHEMA_TEMPLATE_NOT_TRAINING_DATA` | ⚠️ Unverified (template only) |
| 6 | `"Kusheh! Aw di bodi?"` | `datasets/krio/cleaned/example_krio_text_train.jsonl`, `benchmarks/krio/held_out_eval_schema.jsonl` | Multiple templates | Replaced in test fixtures with `TEST_FIXTURE_*`; templates marked not training data | 🔄 Replaced in tests; ⚠️ template only |
| 7 | `"Kusheh, aw di bodi? A wel tenki."` | `preprocessing/krio_normalizer.py` | Comment example | Linguistic comment example | ⚠️ Needs native review |
| 8 | `"Gud mɔning"` | `tests/test_schema.py` | Translation test fixture | Replaced with `TEST_FIXTURE_*` | 🔄 Replaced |
| 9 | `"Kusheh"` | `tests/test_schema.py`, `tests/test_speech_and_training.py` | Test fixture | Replaced with `TEST_FIXTURE_*` | 🔄 Replaced |
| 10 | `"Mi de go maket."` | `tests/test_schema.py`, `tests/test_normalizer.py`, `tests/test_api.py` | Test fixtures | Replaced with `TEST_FIXTURE_*` | 🔄 Replaced |
| 11 | `"Wetin gud taim fɔ plant kasada?"` | `src/sierraleone_ai/api/app.py`, `documentation/API_DOCUMENTATION.md` | API request sample | Replaced with `TEST_FIXTURE_PROMPT_001` | 🔄 Replaced |

---

## Action Plan

**Tests and code (complete):** All Krio fixtures in `tests/` and `src/` are replaced with neutral English or opaque placeholders (`"TEST_FIXTURE_TEXT_001"`, `"FIXTURE_EN_001"`).

**Data pipeline (enforced):** The example JSONL files in `datasets/krio/` and `benchmarks/krio/` are marked `SCHEMA_TEMPLATE_NOT_TRAINING_DATA` — they will never be loaded as training data (enforced by the loader's eligibility gate, manifest build, and training guard).
