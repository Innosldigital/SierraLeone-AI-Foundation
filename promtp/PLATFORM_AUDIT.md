# PLATFORM_AUDIT.md — Sierra Leone AI Foundation

**Audited archive:** `SierraLeone-AI-Foundation_1.zip` (package `sierraleone_ai` 0.1.0, git `main` @ `bae83de` + 8 whitespace-only uncommitted edits)
**Audit date:** 2026-11-11
**Method:** every tracked file was read in full (≈7,100 lines of Python/config/JSON + all docs). Findings tagged **[E]** were additionally *executed* in isolation; **[W]** were checked against an official/primary web source; **[R]** were verified by reading the code; **[U]** are *unverified risks* and are labelled as such.

---

## 0. Read this first — scope and limits

### 0.1 What this repository actually is
A **Python data-governance and ML-scaffolding package** (`src/sierraleone_ai`) plus a **FastAPI stub**. It is **not** a full platform. The following parts of your brief **do not exist in this repository**, so there was nothing to audit there:

| Area in your brief | State in the repo |
|---|---|
| Frontend / user interfaces | **None.** (No web or mobile UI. The CAFP Farmer App is a separate project.) |
| Database | **None.** (Data = JSONL files in git; no DB, no migrations.) |
| Authentication / authorisation | **None** (deliberately deferred; all model endpoints return 503). |
| Third-party integrations | **None** (no calls to Hugging Face, Khaya, Google, etc. except a tokenizer download helper). |
| AI features | **None operational.** No model, no weights, no training loop; all training functions raise `NotImplementedError` or return a dry-run plan. |

I did not invent findings for these. They appear in `PROJECT_ROADMAP.md` as *missing components*.

### 0.2 What I could NOT do
- **I could not run `pytest`, `ruff` or `mypy`.** The sandbox has no network (`pip install` fails) and the bundled `.venv` is a Windows/Python-3.15 environment with compiled `.pyd` files. **"All tests pass" is therefore unverified by me.** Evidence only from your side: `.pytest_cache/v/cache/lastfailed` is `{}` (no failures recorded in your last run), 174 test node IDs exist. `.coverage` is stale (09:41; predates `tokenizer_audit.py`).
- What I *did* run: `python -m compileall` (clean), an AST scan for unused imports (clean), and **isolated executions** of the loader gate, readiness counter, benchmark extractor and split allocator with stand-ins for the missing libraries (results quoted below).

### 0.3 Severity scale
**Critical** – defeats a core safety/rights control or can silently corrupt a result. **High** – real defect or false claim with significant impact. **Medium** – real defect/gap with limited blast radius or latent. **Low** – hygiene, polish, hardening.
"Latent" = harmless today (no real data exists) but will bite as soon as real data arrives.

---

## 1. Verdict in one paragraph

The first round of fixes (SLAI-001…032) was **largely real**: the licence is complete, the SPDX licence format is correct, strict loading with line numbers exists, the split allocator is correct, CI and a clean-archive script exist, and the registry no longer claims Aya-23 is African. **But the central promise — "a data-readiness gate that cannot be fooled" — is still not true**: readiness counts any record regardless of consent/validation/licence/split, the loader gate has permissive defaults, consent *scopes* are only enforced for TTS, and the held-out-benchmark guard silently ignores the benchmark template's own `prompt` field. In addition, the model registry contains **one false factual claim** (NLLB "includes Krio") that contradicts the official model card. None of this is dangerous *today* because there is no data and no model — the right moment to fix it is now, before data arrives.

---

## 2. What works (verified)

| # | Verified working | How |
|---|---|---|
| W1 | `LICENCE` is now the complete Apache-2.0 text (202 lines, "END OF TERMS" + appendix) | [R] |
| W2 | `pyproject.toml` uses SPDX `license = "Apache-2.0"`, `setuptools>=77` | [R] |
| W3 | `deterministic_split` largest-remainder allocation is correct: n=3→1/1/1, 4→2/1/1, 7→5/1/1, 100→80/10/10; n=2 raises a clear error | [E] |
| W4 | Template files (`example_*`, `SCHEMA_TEMPLATE…` sources) are excluded from the readiness count and loaders | [R][E] |
| W5 | Strict loader raises `DataValidationError` with file/line; preview mode is labelled; quarantine output exists | [R] |
| W6 | `/health/live` vs `/health/ready` split; per-call `HTTPException` (no shared instance); `/transcribe` checks type/size/empty/magic bytes and cleans temp files | [R] |
| W7 | HF token is a `SecretStr` accepting `HF_TOKEN` or `SLAI_HF_TOKEN` | [R] |
| W8 | Aya-23 entry is now factually correct; "BBC Krio service" removed; docs now say BBC Pidgin ≠ Krio | [R][W] |
| W9 | Tokenizer audit refuses template/`example_*` corpora and small corpora; uses `trust_remote_code=False`; downloads tokenizer only | [R] |
| W10 | No unused imports; every module compiles | [E] |
| W11 | The *clean archive* script produced a ZIP with 87 files, no `.venv`, `.git`, caches or `.env` | [E] |
| W12 | Git history exists (11 commits), remote `origin` is GitHub | [R] |
| W13 | Source-rights register exists with every external source `UNREVIEWED`; loader blocks listed unreviewed sources | [R] |
| W14 | The DONDO ASR entry in the registry is **real** (arXiv 2607.21540, Khaya AI, Apache-2.0; multilingual `kri-men-tem-en` in the paper's Sierra Leone row) | [W] |

---

## 3. Findings

Format: **ID — Title** · Severity · Evidence · File/location · Explanation · Recommended solution (details in `IMPLEMENTATION_FIXES.md`).

### 3.1 Critical

**PA-001 — Readiness counts ineligible records** · Critical (latent) · [E][R]
`src/sierraleone_ai/training/guard.py:169-211` → `loader.py::_count_real_records` (≈L648-668).
`_count_real_records` skips only template sources, `rejected` status and unreviewed-register sources. It ignores consent, validation status, licence, `training_permitted`, split, duplicates and schema validity, and uses the *non-strict* JSON reader.
*Executed:* 3 records that are `unvalidated`, `contributor_consent=false`, `licence=unknown`, `split=test` plus 1 `machine_generated` record → **counted as 4 "real records"**. So 50,000 unusable rows would turn `cleaned_text` green.
*Also:* `cleaned` + `validated` are summed (the same record in both directories is double-counted); `tts_audio` looks in `datasets/krio/tts/`, a directory that does not exist in the layout, so it is always 0.
**Fix:** compute readiness from the *same* eligibility policy that the training loaders use, per task (PA-002), on schema-validated records, de-duplicated by `record_id`.

**PA-002 — Loader gate evaluates raw dicts with permissive defaults; no re-check after validation** · Critical (latent) · [E][R]
`loader.py::_is_eligible` (≈L198-255) and `load_jsonl_records` (≈L456-566).
Defaults used when a key is absent: `licence="CC-BY-4.0"`, `training_permitted=True`, `status="unvalidated"`, `split="train"`. The pydantic schema's own defaults are the opposite (`licence=UNKNOWN`, which forces `training_permitted=False`). The gate runs on the raw dict *before* parsing and nothing re-checks the parsed model.
*Executed:* a record with **no `licence`, no `consent_scope`, no `training_permitted`** but `contributor_consent=true` → `(True, '')` — **admitted**, and then parsed into a model whose `licence` is `unknown` and `training_permitted=False`.
**Fix:** parse first (schema), then evaluate eligibility on the *validated model*; one policy object; fail closed on missing rights fields.

### 3.2 High

**PA-003 — Consent *scope* is not enforced for text/translation/conversation training; legacy alias widens ASR consent** · High · [E][R]
`loader.py::_is_eligible` checks only the boolean `contributor_consent`; `consent_scope` is read only by `export_for_release` and the TTS/ASR schema validators.
*Executed:* a record with `licence=proprietary` and `consent_scope=["voice_synthesis"]` only → **admitted into a text-training load**.
`schema.py::ConsentScope` keeps legacy aliases (`training`, `evaluation`, `publication`, `commercial`) and `VerifiedSpeechTranscriptionRecord` accepts the generic `"training"` as ASR consent (`asr_scopes = {"speech_recognition","training"}`), which undermines the "separate, explicit consent" design.
**Fix:** task→required-scope map enforced in the policy; remove legacy aliases after a one-time migration.

**PA-004 — Held-out benchmark protection misses the benchmark's own fields** · High · [E][R]
`evaluation/benchmarks.py::_extract_texts_from_jsonl` (L298-318); template `benchmarks/krio/held_out_eval_schema.jsonl`.
Fingerprints are extracted only from `text, source_text, target_text, question, answer, transcript`. The shipped benchmark template uses **`prompt`** (and conversation records use `turns[].text`).
*Executed:* a benchmark row with `prompt` and a conversation row with `turns` → **0 texts extracted**. A native-speaker panel that follows the template would yield an *empty* fingerprint set: `assert_held_out_integrity` returns False (training blocked — fail-closed, good) but the contamination check can never work; conversation training data overlapping a conversation benchmark is also invisible (`loader.py` candidate fields exclude `turns`). There is also no pydantic model for a benchmark record, no versioning, and "integrity" = "≥ 1 fingerprint" (no hash manifest, so edits to the benchmark are undetectable).
**Fix:** `BenchmarkItem` schema; fingerprint every text-bearing field generically; benchmark manifest with SHA-256 + count + version.

**PA-005 — Model registry contains a false claim and a wrong entry** · High · [W][R]
`models/registry.py`:
1. **NLLB-200 "Includes Krio (`kri_Latn`)"** (L320-336). **False.** The official model card's `language_details` list (…`kor_Hang, kmr_Latn, lao_Laoo`…) contains no `kri_Latn`. The same card says the model is "not released for production deployment", licence CC-BY-NC-4.0. This repeats exactly the kind of unverified claim the previous audit removed for Aya-23.
2. **"Tiny Aya (Aya-Expanse variants)" points to `CohereForAI/aya-expanse-8b`** (L257-277). Tiny Aya (Feb 2026, ~3.35B) and Aya Expanse 8B are different model families; the entry's name, parameter range and repo do not describe one thing.
3. `multilingual` flags look inverted/arbitrary (Gemma-3 1B `True`, Gemma-3 4B/12B `False`, Llama-3.1-8B `False`). [U — verify on model cards]
4. All `verified_on=None`, yet entries state specifics (VRAM, licences in parentheses) from memory-level knowledge.
**Fix:** correct/remove entry 1–2; make `source_url` + `verified_on` mandatory via a CI check; move the registry to data (YAML/JSON) with a schema.

**PA-006 — Real data and the held-out benchmark would be committed to git and shipped in "clean" ZIPs** · High (latent) · [R][E]
`.gitignore` ignores only `datasets/*/raw/*`, `datasets/*/audio/*`, `quarantine/`, binaries. `cleaned/`, `validated/`, `translations/`, `conversations/` **and `benchmarks/`** are tracked by default. `scripts/make_source_archive.py::should_include_path` excludes only `raw/` and `audio/` under `datasets/`. The git remote is `https://github.com/Morie55/SierraLeone-AI-Foundation.git` (visibility unknown). Consequences once real data exists: consented-but-private text, unreviewed-rights text and the **held-out benchmark** go to GitHub and into any archive you share (including archives you upload to AI tools — which would contaminate the benchmark).
**Fix:** keep real data and benchmark answers **outside git** (private store); commit schemas, manifests and hashes only; extend `.gitignore` and the archive script to a *deny-by-default* rule for everything under `datasets/**` except templates, `.gitkeep` and `source_register.csv`, and for `benchmarks/**` except templates.

### 3.3 Medium

**PA-007 — Quarantine report files are counted as data** · Medium · [E]
`dataset_summary()` / readiness glob `*.jsonl` in each subfolder; `datasets/krio/quarantine/<name>.issues.jsonl` lines have no `metadata`, so `_count_real_records` counts each issue row as a record (*executed → 1*). **Fix:** skip `quarantine/`, `splits/`, `*.issues.jsonl`; count only schema-valid records.

**PA-008 — Source-rights register is a deny-list on exact names** · Medium · [R]
`_is_eligible` blocks only when `source in source_register` and decision ≠ APPROVED. A record whose `source` is spelled differently (e.g. `"Jaward/Krio-Corpus"` vs register key `jaward_krio_corpus`) or is simply not listed is admitted on its own say-so. **Fix:** allow-list: every non-template `source` must exist in the register with `APPROVED`.

**PA-009 — Robustness: unguarded gate and BOM files** · Medium · [E]
(a) `_is_eligible` is called outside the `try`; `metadata` that is not an object raises `AttributeError` (*executed*), so strict mode fails with an unrelated error instead of `DataValidationError` with a line number. (b) Files saved as UTF-8-with-BOM (typical for Windows editors — the repo author works on Windows) fail on line 1: *executed → "Unexpected UTF-8 BOM"*. **Fix:** guard + record as `LoadIssue`; open with `encoding="utf-8-sig"`.

**PA-010 — Training approval is self-attested and not bound to what trains** · Medium · [R]
`training/guard.py`. The manifest's hashes prove files are unchanged, but nothing checks that (a) `manifest.base_model_id` equals `models/selected_base_model.json`, (b) `manifest.task` matches the requested task, (c) the manifest's files are the files the config will read, (d) the approver is anyone but the author. The dry-run "plan" has no hardware/GPU, cost or tokenizer-compatibility information, although the docs/changelog describe a "hardware check". It is a useful *accident guard*; it is not an access control — the docs should say so. **Fix:** bind and cross-check; add real plan fields; state the threat model.

**PA-011 — Schema design weaknesses** · Medium · [R]
`datasets/schema.py`:
- `training_permitted` / `redistribution_permitted` **default `True`**; only `UNKNOWN` licences force `False`. A `proprietary` or `custom` record inherits permissive defaults (fail-open).
- `licence` (enum) and `licence_id` (string) can disagree (e.g. `licence=CC-BY-4.0`, `licence_id="MIT"`) with no error; `ALLOWED_LICENCE_IDS` excludes common valid SPDX IDs (e.g. `CC-BY-NC-4.0`).
- `is_native_sl_krio_speaker` is Krio-specific although Mende/Temne etc. are in scope.
- No `contributor_id`, so "independent" reviewers cannot be verified to differ from the contributor/transcriber.
- `split` defaults to `TRAIN`: a benchmark record whose author forgot `split` is a training record.
- `ReviewerSignoff` has no `notes` (the changelog says it does).
- `AudioRecord.verify_audio_file` reads the whole file into memory and is never called by the TTS/ASR checks.

**PA-012 — API upload handling** · Medium · [R] (framework behaviour = [U])
`api/app.py`: (a) `await file.read(max+1)` caps *what the handler reads*, but Starlette parses and spools the multipart body before the handler runs, so oversized uploads are still received and stored; enforce the limit at the proxy / via a `Content-Length` check. [U — verify on your Starlette version] (b) Accepted types omit `audio/mp4`, `audio/aac`, `audio/x-m4a`, `audio/3gpp`, `audio/amr` and magic bytes omit `ftyp`/AAC/AMR — typical phone recordings would be rejected with 415/400. (c) The magic check `RIFF` also matches non-WAV RIFF files (WebP/AVI); check bytes 8:12 == `WAVE`. (d) The user's voice is written to a temp file even though the endpoint always returns 503.

**PA-013 — API validation/readiness logic** · Medium · [R]
`/translate` accepts `english→english` and non-English↔non-English pairs (docs say English↔SL); lines 210-215 repeat the same validation four times; `GenerateRequest.language`/`TranslateRequest.*` are free `str` while `SynthesizeRequest.language` is the enum (inconsistent 422 shapes). `/health/ready` is derived from a **static Python dict** (`status: "operational"`), so flipping that string would report ready with no model loaded. **Fix:** derive readiness from a model-loader state.

**PA-014 — Configuration contradictions and dead settings** · Medium · [R][E]
- `.env.example` sets `SLAI_DATA_DIR=datasets` (a *relative* path). Copying it to `.env` overrides the anchored default and re-introduces the CWD-dependence that SLAI-010 removed; `env_file=".env"` is also CWD-relative.
- `.env.example` sets `SLAI_API_HOST=0.0.0.0` (code default is `127.0.0.1`).
- `api_host`, `api_port`, `api_debug`, `logs_dir`, and `hf_token` are **never used** anywhere in `src/` *(grep → no uses)*. `settings.hf_token` is not passed to the tokenizer loader, so gated tokenizers (Llama, Gemma) will fail unless the user logs in separately.
- `cors_origins: list[str]` is not documented in `.env.example`/docs; pydantic-settings expects JSON for list env vars; there are no CORS tests; the default when empty is hard-coded `["http://localhost:3000"]`.

**PA-015 — Reproducibility and CI gaps** · Medium · [R]
- `constraints.txt` is titled "Pinned…" but contains **ranges**, not pins, and **CI never uses it** (`pip install -e ".[dev,eval]"`).
- CI runs Linux only; your dev environment is Windows + Python 3.15.2; CI matrix is 3.10/3.12/3.13. Path/encoding bugs (e.g. PA-009 BOM) won't be caught.
- No `ruff format --check`; `.pre-commit-config.yaml` pins ruff `v0.1.9` while your cache shows ruff 0.17.4; third-party actions are tag-pinned, no `permissions:` block; no optional-extra import check; no coverage threshold.
- The "no Krio text in tests" grep covers six substrings in `tests/` only (e.g. `bo,` would match innocent English), not docs/src/datasets.

**PA-016 — CLI gives false assurance** · Medium · [R]
`slai validate-file` ("Validate… against Pydantic schema rules") only checks that each line is JSON containing a `metadata` key — `{"metadata": {}}` is "valid". `slai status` rows 1–2 are hard-coded strings ("implemented … gates active") with no runtime check; "implemented" is also used for "all readiness thresholds met".

**PA-017 — Documentation no longer matches the code** · Medium · [R]
- `DATASET_FORMAT.md` examples are **invalid against the current schema**: every example is `validated` + `contributor_consent: true` but has no `reviews` (≥2 native approvals) and no `consent_scope`; none of the new fields are documented.
- `CLAUDE.md`: test counts contradict (35 / 110+ / 110 passing vs **168** test functions); instructs agents to raise `_MODEL_NOT_READY` (now `_model_not_ready()`); "Adding a New Language" says edit `SUPPORTED_LANGUAGES` (now *derived*; editing it does nothing) and "no changes to schemas"; the `Language` enum, `LANGUAGE_REGISTRY` and Krio-specific loaders (`load_krio_text`, `krio_data_dir`) all need changes; "6 endpoints" (actual: 8); Rule 2 asserts unreviewed linguistic claims as absolute (see PA-019).
- `API_DOCUMENTATION.md`: omits `/health/live`, `/health/ready`, upload error codes 413/415/422; says validation errors are 400 (FastAPI uses 422); recommends `--host 0.0.0.0 --workers 4` "for production" with no auth, rate limit or TLS; contains an unmarked Krio sample.
- `DEPENDENCIES_AND_LICENCES.md`: lists `jsonschema` as a core dependency (not in `pyproject.toml`); puts `sacrebleu` under *training* (it is the `eval` extra); cross-references "CLAUDE.md Rule 5 (document every dependency)" but Rule 5 is the secrets rule.
- `SYSTEM_ARCHITECTURE.md`: "model registry: per-language model entries" — it is per-task.
- `docs/NATIVE_REVIEW_QUEUE.md`: rows 6, 8, 9, 10 say test files contain these strings; they no longer do.
- `IMPLEMENTATION_CHANGELOG.md` inaccuracies: finding IDs are renumbered and no longer match the audit; says `ReviewerSignoff.notes` exists (it doesn't); lists example files `example_krio_text.jsonl` etc. that do not exist (actual names differ); says SLAI-011 added a Pydantic validator (it is a dataclass with an entry-point check); says `hf_token` is `SecretStr | None` (it is `SecretStr` with default `""`).
- `pyproject.toml [project.urls]` point to `github.com/cafp-sierraLeone/…` while `origin` is `github.com/Morie55/…`.
- README structure tree omits `speech/`; `copy .env.example .env` is Windows-only; README/pipeline still frame **continued pre-training first** as "the" approach (see PA-024).

**PA-018 — Evaluation metrics are non-standard** · Medium · [R]
`evaluation/metrics.py`: `chrf_score` counts spaces as characters, has no word n-grams (docstrings say "chrF/chrF++"), so values are not comparable with published chrF/chrF++ (`sacrebleu` is declared in the `eval` extra but never used); short-hypothesis BLEU smoothing makes scores incomparable with standard BLEU; `_edit_distance` allocates an O(n·m) matrix (a 5 k-character CER needs ~25 M cells). Fine as unit-test helpers; not fit for reporting.

### 3.4 Low

**PA-019 — Normaliser and Krio-claims** · Low · [R]
`krio_normalizer.py`: `\r` survives (whitespace regex is `[ \t]+`; control-char strip excludes `\r`), so CRLF text from Windows keeps stray `\r`; the "overlong" check uses the *pre-normalisation* length; the cross-creole marker list is explicitly unreviewed (good) but `CLAUDE.md` Rule 2 states the same distinctions as *absolute* facts ("not `abeg`, `chale`, `shey`") — unverified linguistic claims in an instruction file for AI agents can cause them to reject valid Krio. Treat Rule 2's examples as hypotheses pending native review. `comot` is flagged though the code's own note calls it a spelling variant.

**PA-020 — Tokenizer audit** · Low · [R]
`json.loads` is unguarded (malformed line → raw exception); `partially_validated` rows qualify by default; the report does not record tokenizer revision/commit (not reproducible); `settings.hf_token` unused (PA-014).

**PA-021 — Code structure** · Low · [R]
`ReadinessItem` carries three alias properties (`name`, `ready`, `count`) "for tests"; `benchmarks.py` imports the private `_TEMPLATE_PREFIXES` from `loader.py` while `loader.py` imports `benchmarks` lazily (circular-import workaround) and `tokenizer_audit.py` redefines the tuple (three copies); `LoadReport` defines `__eq__` so it is unhashable and compares equal to a `list`; `stats.py` column says "Files found" but shows record counts; whole files are loaded into memory; the source register and benchmark fingerprints are re-read on every load.

**PA-022 — Split manager** · Low · [R]
Manifest is written non-atomically (a crash can corrupt the file that the no-leakage guarantee depends on); splitting is by *speaker count*, not audio duration, so the test split's hours can vary widely; incremental additions can drift from target ratios.

**PA-023 — Speech helpers** · Low · [R]
TTS gate trusts metadata (does not call `verify_audio_file`); environment compare is case-sensitive; `ASRTrainingConfig.require_dual_verified_transcripts/min_duration/max_duration` are never enforced; default `base_model="openai/whisper-small"` is hard-coded while an open Sierra-Leone ASR baseline now exists (see roadmap).

**PA-024 — Training recipe defaults** · Low (design) · [R][U]
CPT-first, 3 epochs, `learning_rate=2e-5` with QLoRA (2e-5 is a full-fine-tune-scale value; LoRA runs usually use higher — **[U] verify on a pilot**), LoRA `target_modules` use Llama-style names (not valid for every architecture), `modules_to_save` empty. With ≈1 M tokens of Krio expected versus 10–26 B tokens used by published African-language CPT studies, CPT-first is the wrong default.

**PA-025 — Test quality** · Low · [R]
`pytest` ignores `DeprecationWarning` from starlette/pydantic; `test_registry.py:89` silently `skip`s if Aya-23 is missing; fixtures always set `licence` and `consent_scope=["training"]` (legacy alias), so PA-002/PA-003 are not covered; no CORS tests; the readiness tests never exercise a *passing* threshold; no Windows run.

---

## 4. Critical evaluation of the existing design

Per your instruction, I did not assume the architecture is right. Decisions:

| Component | Assessment | Decision |
|---|---|---|
| **Eligibility logic** — implemented in 4 places (`_is_eligible` on dicts, schema validators, `_count_real_records`, `export_for_release`) with different rules | Unsuitable: the rules drift (that is PA-001/002/003) | **Redesign** into one policy over validated models |
| **Readiness = scan files in folders** | Unsuitable: counts files/lines, not trustworthy data | **Replace** with a generated, hashed *dataset manifest* that the guard consumes |
| **Real data as JSONL inside the git repo** | Unsuitable for consented/private/rights-restricted data and a secret benchmark (PA-006) | **Replace**: private store + manifests in git |
| **Model registry as Python source** | Unsuitable: claims can't be audited; one false claim already shipped (PA-005) | **Redesign** as data + schema + CI verification |
| **Two language sources** (`Language` enum, `LANGUAGE_REGISTRY`) and Krio-hard-coded loaders | Unsuitable for a multilingual platform | **Redesign**: one registry; `load_records(language, kind, split)` |
| **Training guard = env flag + JSON attestation** | Reasonable *accident* guard; not access control | **Keep & fix** (bind to selection/datasets; document threat model) |
| **CPT-first pipeline** | Unsuited to ≈1 M tokens | **Replace default** with baseline-first → SFT/LoRA → CPT only if justified |
| **Home-grown BLEU/chrF** | Unsuitable for reporting | **Replace** with `sacrebleu`; keep toy versions for tests only |
| **API with model code in the web process** (future) | Not yet built | **Plan** a separate inference service (see roadmap) |
| **Static "operational" flag** | Unsuitable | **Redesign**: readiness from loaded artifact + approval |
| **`validate-file` / `status` CLI** | Misleading | **Fix** (call real loader; compute state) |
| Schema `is_native_sl_krio_speaker` | Krio-only in a multilingual design | **Fix** (rename + migrate) |
| Package layout, Pydantic schemas, strict loader concept, split manager, tokenizer-only audit, source register, consent-scope concept | Sound foundations | **Keep** |

---

## 5. What is missing for a "production-ready platform" (not defects — gaps)
Data-collection tooling (no way yet for contributors to submit consented text/audio); annotation/review workflow; consent store (restricted); private data storage; benchmark with native-reviewed items; inference service + model loader; authentication/rate limiting/observability; deployment (container, CI/CD); user-facing client (CAFP integration); documentation of the threat model. See `PROJECT_ROADMAP.md`.

---

## 6. Summary table

| ID | Sev | Title |
|---|---|---|
| PA-001 | Critical | Readiness counts ineligible records |
| PA-002 | Critical | Loader gate: permissive defaults, no post-parse check |
| PA-003 | High | Consent scope not enforced for text; legacy alias widens ASR consent |
| PA-004 | High | Benchmark guard ignores `prompt` / `turns`; "integrity" is weak |
| PA-005 | High | Registry: false NLLB-Krio claim; wrong Tiny Aya entry |
| PA-006 | High | Real data + benchmark would land in git and shared ZIPs |
| PA-007 | Medium | Quarantine files counted as data |
| PA-008 | Medium | Source register is a deny-list on exact names |
| PA-009 | Medium | Unguarded gate; BOM files fail |
| PA-010 | Medium | Approval manifest not bound; dry-run incomplete |
| PA-011 | Medium | Schema defaults fail-open; Krio-specific field; no contributor id |
| PA-012 | Medium | Upload cap after spooling [U]; phone formats rejected; RIFF check |
| PA-013 | Medium | `/translate` validation; static readiness |
| PA-014 | Medium | `.env.example` contradictions; unused settings; CORS undocumented |
| PA-015 | Medium | Constraints not pins/unused; CI Linux-only; stale pre-commit |
| PA-016 | Medium | `validate-file` and `status` give false assurance |
| PA-017 | Medium | Docs/changelog/CLAUDE.md out of sync with code |
| PA-018 | Medium | Metrics non-standard |
| PA-019 | Low | Normaliser CR handling; unreviewed Krio claims in CLAUDE.md |
| PA-020 | Low | Tokenizer audit robustness/reproducibility |
| PA-021 | Low | Code structure smells |
| PA-022 | Low | Split manager atomicity/duration |
| PA-023 | Low | Speech helper gaps |
| PA-024 | Low | Training recipe defaults |
| PA-025 | Low | Test quality gaps |
