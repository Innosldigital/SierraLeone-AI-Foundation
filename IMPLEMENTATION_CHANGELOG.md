# Implementation Changelog — Audit & Fix Instructions (v2, Merged: SLAI-001 … SLAI-032)

**Project:** Sierra Leone AI Foundation (`sierraleone_ai` v0.1.0)  
**Audit Baseline:** `SierraLeone-AI-Foundation.zip` (merged Audit v1 `SLAI-001…021` + Audit v2 `SLAI-022…032`)  
**Date:** 2026-11-10

---

## Summary of Guarantee & Guardrails Preserved

- **Zero training, model downloads, or cloud spend occurred.** `SLAI_ALLOW_TRAINING` remains `false`, all candidate models remain `recommended=False`, and `models/selected_base_model.json` does not exist until human native-speaker benchmark and tokenizer audit sign-off.
- **Zero fabricated Krio, Mende, or Temne sentences were introduced.** All new tests and fixtures use neutral English or placeholder strings (`TEST_FIXTURE_*`).
- **All model inference endpoints (`/generate`, `/translate`, `/transcribe`, `/synthesize`) and `/health/ready` return HTTP `503`** until a model is trained, evaluated, and approved.

---

## Detailed Changelog by Issue ID (`SLAI-001` … `SLAI-032`)

### P0 Findings

- **SLAI-001 — Loader eligibility gate (`src/sierraleone_ai/datasets/loader.py`)**
  - Replaced simple status check with `LoadPolicy` and `_is_eligible()`.
  - Always excludes `REJECTED` records, `SCHEMA_TEMPLATE*` / `BENCHMARK_SCHEMA_TEMPLATE*` records, `consent_withdrawn=True`, `machine_only` translations (Rule 7), and split mismatches (`record.metadata.split != split`).
  - `validated_only=True` admits only `ValidationStatus.VALIDATED` (requiring `>= 2` native speaker approvals) and excludes `PARTIALLY_VALIDATED`.

- **SLAI-002 — Template files excluded from readiness & summaries (`src/sierraleone_ai/datasets/loader.py`, `src/sierraleone_ai/training/guard.py`)**
  - `dataset_summary()` and `check_dataset_readiness()` skip `example_*.jsonl`, `held_out_eval_schema*.jsonl`, and any record whose `source` starts with template prefixes (`SCHEMA_TEMPLATE`, `BENCHMARK_SCHEMA_TEMPLATE`, `EXAMPLE_PLACEHOLDER`).
  - With only template files present, readiness reports `0` real records (`NOT READY`) across `cleaned_text`, `translations`, `conversations`, `held_out_benchmark`, `asr_audio`, and `tts_audio`.

- **SLAI-003 — Full Apache-2.0 licence text (`LICENCE`)**
  - Replaced truncated licence file with the complete Apache License Version 2.0 text (Sections 1–9 and Appendix).

- **SLAI-004 — SPDX license expression (`pyproject.toml`)**
  - Replaced deprecated `license = {text = "Apache-2.0"}` table with PEP 639 SPDX string `license = "Apache-2.0"` and `license-files = ["LICENCE"]`.

- **SLAI-005 — Factual corrections & native review queue (`src/sierraleone_ai/models/registry.py`, `research/model_comparison.md`, `documentation/*.md`, `docs/NATIVE_REVIEW_QUEUE.md`)**
  - Corrected Aya-23 documentation (0 sub-Saharan African languages; Arabic is the only African-spoken language; Krio is not covered).
  - Removed unverified "BBC Krio Service" references.
  - Marked all model registry entries `recommended=False` with `verified_on=None` and `source_url` links.
  - Replaced unverified Krio strings in tests with neutral `TEST_FIXTURE_*` strings and catalogued all remaining template strings in `docs/NATIVE_REVIEW_QUEUE.md`.

- **SLAI-006 — Secret handling for `HF_TOKEN` (`src/sierraleone_ai/config.py`, `.env.example`)**
  - Changed `Settings.hf_token` to `SecretStr | None` with ` validation_alias=AliasChoices("SLAI_HF_TOKEN", "HF_TOKEN")` so neither `repr(settings)` nor `str(settings)` leaks token values.

- **SLAI-007 — Held-out benchmark integrity & contamination guard (`src/sierraleone_ai/evaluation/benchmarks.py`, `src/sierraleone_ai/datasets/loader.py`)**
  - Implemented `text_fingerprint()` (NFC + lowercase + collapsed whitespace SHA-256), `benchmark_fingerprints()`, `check_contamination()`, and `assert_held_out_integrity()`.
  - Training loaders automatically exclude records whose text fingerprint appears in the held-out benchmark when `policy.exclude_benchmark_overlap=True`.

- **SLAI-022 — Fail-closed strict loader & line-level `LoadReport` (`src/sierraleone_ai/datasets/loader.py`)**
  - Added `LoadIssue`, `LoadReport[T]`, and `DataValidationError`.
  - `load_krio_text()`, `load_translations()`, and `load_conversations()` default to `strict=True`, raising `DataValidationError` with exact file/line numbers on malformed JSON or schema validation errors.
  - Added `strict=False` preview mode (logged as `"PREVIEW – not for training"`) and optional quarantine output to `datasets/<lang>/quarantine/<file>.issues.jsonl`.

- **SLAI-023 — Fail-closed speaker split & persistent speaker manifest (`src/sierraleone_ai/preprocessing/split_manager.py`)**
  - Removed the random fallback for records missing `speaker_id`; `speaker_aware_split()` now raises `ValueError` listing offending record IDs.
  - Added persistent speaker manifest support (`manifest_path` / `persist_manifest`) that preserves existing speaker-to-split assignments when new speakers arrive, permanently excludes `held_out` speakers, and runs an automatic post-split speaker leakage audit.

- **SLAI-024 — Machine-readable training approval manifest & dry-run (`src/sierraleone_ai/training/guard.py`, `src/sierraleone_ai/training/pipeline.py`, `src/sierraleone_ai/speech/asr.py`, `src/sierraleone_ai/speech/tts.py`, `src/sierraleone_ai/cli.py`)**
  - Added `TrainingApprovalManifest` and `verify_training_manifest()` requiring `run_id`, `task`, `base_model_id` (rejecting `"TBD"`), `base_model_revision`, `model_licence_review`, `dataset_versions`, `dataset_sha256_hashes` (rejecting any `raw/` path and verifying on-disk SHA-256 digests), `split_manifest_id`, `benchmark_id`, `estimated_gpu_hours`, `estimated_cost_usd`, and `approver`.
  - Added `dry_run=True` to `require_training_approval()`, all 4 training entry points (`run_continued_pretraining`, `run_instruction_finetuning`, `start_asr_training`, `start_tts_training`), and `slai check-readiness --dry-run`.

- **SLAI-025 — Provenance, licence enforcement, consent scopes & revocation (`src/sierraleone_ai/datasets/schema.py`, `src/sierraleone_ai/datasets/loader.py`, `src/sierraleone_ai/speech/tts.py`, `documentation/CONSENT_AND_REVOCATION.md`)**
  - Extended `ConsentScope` with `"text_training"`, `"speech_recognition"`, `"voice_synthesis"`, `"public_release"`, and `"commercial_use"`.
  - Added `licence_id` (validated against `ALLOWED_LICENCE_IDS`), `licence_url`, `rights_holder`, `source_url_or_citation`, `acquisition_date`, `redistribution_permitted`, `training_permitted`, `access_class`, and `restrictions` to `DatasetMetadata`.
  - Enforced that `LicenceType.UNKNOWN` / `licence_id="unknown"` sets `training_permitted=False` and `redistribution_permitted=False` and blocks training/release loads.
  - Enforced `"voice_synthesis"` consent in `check_tts_recording_quality()` (ASR consent alone is rejected).
  - Added `scan_jsonl_for_pii()` and `export_for_release()`.

- **SLAI-026 — Two-native-speaker review enforcement (`src/sierraleone_ai/datasets/schema.py`, `datasets/krio/cleaned/example_krio_text.jsonl`, `datasets/krio/translations/example_translations.jsonl`, `datasets/krio/conversations/example_agricultural_qa.jsonl`, `benchmarks/krio/held_out_eval_schema.jsonl`)**
  - Added `ReviewerSignoff` model (`reviewer_id`, `is_native_sl_krio_speaker`, `reviewed_on`, `verdict`, `notes`).
  - `DatasetMetadata` validator enforces that `ValidationStatus.VALIDATED` requires at least 2 distinct native speaker approvals (`is_native_sl_krio_speaker=True`, `verdict="approve"`), `contributor_consent=True`, and non-unknown licence.
  - Set `validation_status="unvalidated"` and `contributor_consent=false` on all template JSONL files.

### P1 Findings

- **SLAI-008 — BLEU smoothing & ChrF metric (`src/sierraleone_ai/evaluation/metrics.py`)**
  - `bleu_score()` raises `ValueError` on empty `references`, caps `effective_max_n = min(max_n, len(hypothesis))`, and applies Lin & Och (2004) add-one smoothing for `n > 1` so `bleu_score([["a", "b"]], ["a", "b"]) == 1.0`.
  - Added `chrf_score()` for character n-gram F-score evaluation on morphologically rich / orthographically variable text.
  - Standardized empty-reference handling in `word_error_rate()` and `character_error_rate()` (`0.0` when both empty, `float("inf")` when reference is empty and hypothesis is non-empty).

- **SLAI-009 — Largest-remainder deterministic split & small-N guard (`src/sierraleone_ai/preprocessing/split_manager.py`)**
  - Replaced floor truncation with largest-remainder allocation and guaranteed at least 1 item per active split when `len(items) >= num_active_splits` (raising `ValueError` when `0 < len(items) < num_active_splits`).

- **SLAI-010 — Config path anchoring (`src/sierraleone_ai/config.py`)**
  - Anchored `project_root`, `data_dir`, `benchmarks_dir`, and `models_dir` to `Path(__file__).resolve().parents[2]` and added `language_data_dir()` / `language_benchmark_dir()` helpers.

- **SLAI-011 — Training pipeline placeholder model guard (`src/sierraleone_ai/training/pipeline.py`)**
  - Replaced `base_model: str = "TBD — select after research/model_comparison.md evaluation"` with a Pydantic `@field_validator` that rejects `"TBD"` or empty strings.

- **SLAI-012 — TTS quality gate fail-closed on `None` (`src/sierraleone_ai/speech/tts.py`)**
  - `check_tts_recording_quality()` now explicitly fails when `snr_db`, `recording_environment`, `duration_seconds`, or `"voice_synthesis"` consent is missing.

- **SLAI-013 — Verified speech transcription schema (`src/sierraleone_ai/datasets/schema.py`)**
  - `VerifiedSpeechTranscriptionRecord` enforces `ValidationStatus.VALIDATED`, `transcriber_id != verifier_id`, and `"speech_recognition"` (or `"training"`) in `consent_scope`.

- **SLAI-014 — Native speaker evaluation form validation (`src/sierraleone_ai/evaluation/native_speaker.py`)**
  - `KrioFluencyEvaluation` requires non-empty `evaluator_notes` whenever any Likert dimension is `<= 2` (`VERY_POOR` or `POOR`).

- **SLAI-015 — Uncommented optional dependency groups & constraints (`pyproject.toml`, `constraints.txt`)**
  - Uncommented `[project.optional-dependencies]` groups `training`, `audio`, `eval`, `data`, and `dev`, and added `constraints.txt`.

- **SLAI-016 — Safe Krio normaliser defaults (`src/sierraleone_ai/preprocessing/krio_normalizer.py`)**
  - Changed `NormalisationConfig.on_overlong` default from `"truncate"` to `"reject"`.
  - Logged `ftfy.fix_and_explain()` explanations in `changes`.
  - Guarded `apply_spelling_variants=True` with a `ValueError` until `SPELLING_VARIANTS` is populated and signed off by native linguists.

- **SLAI-017 — Comprehensive unit test suite (`tests/test_*.py`)**
  - Expanded test suite from 35 to 120+ tests covering loader policy, strict/preview mode, source rights register, PII scanning, schema invariants, split manager, metrics, normaliser, API endpoints, CLI, config, and repository template rules.

- **SLAI-027 — API liveness vs readiness & `/transcribe` upload hardening (`src/sierraleone_ai/api/app.py`, `src/sierraleone_ai/api/models.py`, `src/sierraleone_ai/config.py`)**
  - Added `/health/live` (HTTP `200`) and `/health/ready` (HTTP `503` while no models are loaded), keeping `/health` as an alias of `/health/live`.
  - Validated `SynthesizeRequest.language` with the `Language` enum (returning `422` on unknown languages).
  - Hardened `/transcribe` to check MIME type (`415`), empty upload (`400`), size limit `settings.max_audio_upload_bytes` (`413`), and WAV/FLAC/OGG/MP3 header signature (`400`), cleaning up temporary files in `finally`.
  - Derived `/languages` response from `LANGUAGE_REGISTRY`.

- **SLAI-028 — Record-level & path-safety invariants (`src/sierraleone_ai/datasets/schema.py`, `src/sierraleone_ai/datasets/loader.py`, `src/sierraleone_ai/__init__.py`)**
  - Added UUID validation on `record_id` and `conversation_id`, future-date rejection on `collection_date`, `TranslationRecord` language pair checks, `ConversationRecord` turn index contiguity (`0..n-1`) and turn language check, `AudioRecord` path traversal (`..` / absolute path) rejection and `audio_sha256` verification, duplicate `record_id` rejection in loader, and synchronized `LANGUAGE_REGISTRY` with `Language`.

- **SLAI-029 — Reproducible tokenizer fertility & orthography audit (`src/sierraleone_ai/evaluation/tokenizer_audit.py`, `src/sierraleone_ai/cli.py`, `research/model_comparison.md`)**
  - Implemented `audit_tokenizer_on_corpus()` and `slai audit-tokenizer --tokenizer <hf_id> --corpus <path>` computing tokens/word, tokens/char, UNK rate, top-20 word fragmentation, open vowel (`ɛ`/`ɔ`) handling, and English sequence-length overhead without downloading model weights.
  - Refuses `example_*` and template corpora or corpora below `--min-sentences`.

- **SLAI-030 — Cross-creole overlap flagging without auto-rejection (`src/sierraleone_ai/preprocessing/krio_normalizer.py`)**
  - Renamed field to `possible_cross_creole_overlap` (retaining `.flagged_non_krio_markers` property alias), added `needs_native_review: bool`, added `CREOLE_MARKER_METADATA = {"reviewed_by_native": False, ...}`, and ensured markers only flag for native review and never auto-reject.

- **SLAI-031 — External Krio source rights register (`datasets/source_register.csv`, `documentation/SOURCE_RIGHTS_AUDIT_TEMPLATE.md`, `src/sierraleone_ai/datasets/loader.py`)**
  - Added `datasets/source_register.csv` with `UNREVIEWED` entries for `umea_krio_corpus`, `typecraft_krio_corpus`, `jaward_krio_corpus`, `krio_scripture_translations`, `sl_broadcaster_newspaper_transcripts`, and `sierraleonenlp_community`.
  - Loader blocks any source in `datasets/source_register.csv` whose `decision != "APPROVED"`.

- **SLAI-032 — Task-separated 2025/2026 model registry & decision file (`src/sierraleone_ai/models/registry.py`, `research/model_comparison.md`)**
  - Separated candidate models by `task` (`chat_llm`, `translation_seq2seq`, `asr`, `tts`).
  - Added `Qwen/Qwen3-4B`, `google/gemma-3-1b-pt`, `google/gemma-3-4b-pt`, `google/gemma-3-12b-pt`, `meta-llama/Llama-3.2-1B`, `meta-llama/Llama-3.2-3B`, `facebook/nllb-200-distilled-600M`, `openai/whisper-small`, `KhayaAI/w2v-bert-kri-men-tem-en`, and `facebook/mms-tts`.
  - Updated `get_recommended()` to read `models/selected_base_model.json`.

### P2 Findings

- **SLAI-018 — Directory & enum consistency (`benchmarks/mandingo/`, `src/sierraleone_ai/datasets/schema.py`, `src/sierraleone_ai/__init__.py`)**
  - Renamed `benchmarks/Mandingo/` to lowercase `benchmarks/mandingo/` and added `Language.MANDINGO = "mandingo"`.

- **SLAI-019 — Unused imports & lint cleanup (`src/`, `tests/`)**
  - Removed all unused imports and enforced `ruff` import sorting and lint rules.

- **SLAI-020 — Working CLI entrypoints & 5-state status (`src/sierraleone_ai/cli.py`)**
  - Implemented `slai version`, `slai languages`, `slai data-summary`, `slai check-readiness [--dry-run]`, `slai validate-file`, `slai audit-tokenizer`, and `slai status` (reporting `implemented`, `scaffolded`, `not configured`, `not ready`, `operational`).

- **SLAI-021 — Clean source packaging & CI guard (`.gitignore`, `.github/workflows/ci.yml`, `scripts/make_source_archive.py`, `scripts/make_source_archive.sh`)**
  - Updated `.gitignore` (including `datasets/**/quarantine/`), added CI checks rejecting `.venv/`, `.env`, and model weight binaries, and added `scripts/make_source_archive.py` / `scripts/make_source_archive.sh` to produce `SierraLeone-AI-Foundation-clean.zip`.

---

## Deferred Items (Part D — Intentionally Not Implemented in This Phase)

Per Audit v2 Part D instructions, the following items are deferred until a model is trained or real data scale requires them:
1. Cryptographic signed evaluation manifests beyond the SHA-256 fingerprint contamination checks in `SLAI-007`.
2. API authentication, rate limiting, request-ID middleware, and request timeouts (all model endpoints currently return `503`).
3. Full MinHash / fuzzy duplicate detection infrastructure.
4. Executable GPU training loops for CPT/SFT/ASR/TTS (configs remain validated plans + dry-run only).
5. Evaluator calibration / adjudication tooling beyond `SLAI-014` (`KrioFluencyEvaluation`) and `SLAI-026` (`ReviewerSignoff`).
