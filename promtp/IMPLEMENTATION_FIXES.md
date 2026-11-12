# IMPLEMENTATION_FIXES.md — Sierra Leone AI Foundation

Companion to `PLATFORM_AUDIT.md` (finding IDs `PA-001 … PA-025`). Written to be handed to a coding agent (Antigravity) **and** to be readable by you.

> Code below was **not executed against your full test-suite** (no network in the audit sandbox). Treat every snippet as a reviewed starting point: after applying each fix run `pytest -q`, `ruff check .`, `mypy src`. Where a behaviour depends on a library version it says **[verify]**.

---

## 0. Ground rules for whoever implements this

1. Work only in this repo. Do not touch the CAFP Farmer App.
2. **No new Krio/Mende/Temne text anywhere** (tests use `TEST_FIXTURE_*`).
3. No training, no model-weight downloads, no cloud spend. Tokenizer downloads only when explicitly asked.
4. Do not weaken a safeguard to make a test pass; fix the test.
5. One commit per `PA-` ID; commit message `PA-00X: …`. Run tests before each commit.
6. Every fix that changes behaviour gets a regression test that **fails on the old code** (the audit gives the failing case).
7. After all fixes, update `IMPLEMENTATION_CHANGELOG.md` using the **PA-** IDs (and map old SLAI IDs to them).

### Recommended order
1. PA-006 (stop data leakage paths) → 2. PA-011 (schema defaults) → 3. PA-002/003/008/009 (one eligibility policy) → 4. PA-001/007 (readiness) → 5. PA-004 (benchmark) → 6. PA-005 (registry truth) → 7. PA-010 → 8. PA-012/013/014 → 9. PA-015/016/017 → 10. the Low items.

### Dependencies introduced
| Need | Package | Extra | Why |
|---|---|---|---|
| Standard metrics | `sacrebleu>=2.3` (already in `eval`) | `eval` | PA-018 |
| WER/CER (optional) | `jiwer` **[verify current release]** | `eval` | PA-018 (or keep the simple versions for tests only) |
| Lockfile | `uv` (you already use uv's Python) | dev tool | PA-015 |
| YAML registry | `pyyaml` | core | PA-005 (or use JSON and add nothing) |

No database is introduced. **Database migrations: none required** (the repo has no database). The only "migration" is a one-off JSONL schema migration (§PA-011) — there are zero real records today, so it is trivial now and expensive later.

---

## PA-006 — Keep real data and the benchmark out of git and out of shared ZIPs

**Goal:** deny-by-default for anything that is data.

1. `.gitignore` — append:
   ```gitignore
   # Deny-by-default for data: only templates, keepfiles, register and manifests are committed
   datasets/**
   !datasets/
   !datasets/**/
   !datasets/**/.gitkeep
   !datasets/**/example_*.jsonl
   !datasets/source_register.csv
   !datasets/manifests/
   !datasets/manifests/*.manifest.json

   benchmarks/**
   !benchmarks/
   !benchmarks/**/
   !benchmarks/**/.gitkeep
   !benchmarks/**/*schema*.jsonl
   !benchmarks/manifests/
   !benchmarks/manifests/*.manifest.json
   *.zip
   ```
   Then run `git rm -r --cached datasets benchmarks` **only if** real data was ever added, and re-add the allowed files. (Today only templates are tracked.)
2. `scripts/make_source_archive.py::should_include_path` — replace the `raw/audio` rule with an allow-list:
   ```python
   def _data_allowed(rel: PurePosixPath) -> bool:
       parts = rel.parts
       name = parts[-1]
       if parts[0] == "datasets":
           return (name == ".gitkeep" or name.startswith("example_")
                   or name == "source_register.csv" or "manifests" in parts)
       if parts[0] == "benchmarks":
           return (name == ".gitkeep" or "schema" in name or "manifests" in parts)
       return True
   ...
   if not _data_allowed(rel_path):
       return False
   ```
3. **Store real data privately.** Pick one (see roadmap §4): private Hugging Face dataset repos, or DVC + S3-compatible/R2 storage, or an encrypted drive for the pilot. The repo keeps only `datasets/manifests/<name>-vN.manifest.json` (file names, record counts, SHA-256, schema version).
4. **Tests:** `tests/test_archive.py` — create a temp tree containing `datasets/krio/cleaned/krio_text_train.jsonl`, `benchmarks/krio/real.jsonl`, `.env`, `.venv/x`; assert none are included, while `example_*.jsonl`, `.gitkeep`, `source_register.csv` are.
5. **CI:** add a step failing if `git ls-files datasets benchmarks` contains anything outside the allow-list.

---

## PA-011 — Schema: fail-closed defaults, generic reviewer field, contributor id

**File:** `src/sierraleone_ai/datasets/schema.py`

Changes:
```python
class DatasetMetadata(BaseModel):
    schema_version: Literal[2] = 2                 # NEW – enables migrations
    ...
    split: DataSplit                               # REQUIRED – remove default TRAIN
    contributor_id: str | None = None              # anonymised; needed to prove reviewer independence
    redistribution_permitted: bool = False         # was True
    training_permitted: bool = False               # was True
    # licence_id becomes the single source of truth; keep `licence` as a derived read-only alias
```
- **Licence consistency:** if both `licence` and `licence_id` are given and differ → `ValueError`. Extend `ALLOWED_LICENCE_IDS` with `"CC-BY-NC-4.0"`, `"CC-BY-NC-SA-4.0"`, `"ODbL-1.0"`, `"MPL-2.0"`, `"BSD-3-Clause"` (or accept any SPDX id from a maintained list **[verify]**), and require `licence_url` when `licence_id in {"custom","proprietary"}`.
- **Explicit permission, not default:** a record is training-eligible only if `training_permitted is True` **and** the licence is not unknown. Data collectors must set it deliberately.
- **Reviewer independence:** in the VALIDATED validator also require `meta.contributor_id not in {r.reviewer_id for r in approvals}` when `contributor_id` is set.
- **Rename** `ReviewerSignoff.is_native_sl_krio_speaker` → `is_native_speaker_of_record_language` with `validation_alias=AliasChoices(...old...)` so old rows still load; same for `KrioFluencyEvaluation.evaluator_native_krio` (add `evaluator_native_speaker` alias). Add optional `notes: str | None`.
- **Consent scopes:** delete the legacy aliases from `ConsentScope` *after* running the migration below. In `VerifiedSpeechTranscriptionRecord`, require `"speech_recognition"` only.
- `AudioRecord.verify_audio_file`: hash in chunks:
  ```python
  h = hashlib.sha256()
  with target.open("rb") as f:
      for chunk in iter(lambda: f.read(1 << 20), b""):
          h.update(chunk)
  ```

**One-off migration (no real data yet, so this is a script + test, not a project):** `scripts/migrate_schema_v1_to_v2.py` reads each JSONL, maps `consent_scope` aliases (`training→text_training`, `publication→public_release`, `commercial→commercial_use`; `evaluation` is dropped), sets `schema_version=2`, and **refuses** to guess `training_permitted` (writes `false` and a note) — a human must set it.

**Tests (fail on old code):** a CUSTOM-licence record with no flags is not training-eligible; `licence=CC-BY-4.0, licence_id="MIT"` raises; benchmark record without `split` raises; reviewer == contributor does not count as an approval.

---

## PA-002 / PA-003 / PA-008 / PA-009 — One eligibility policy on validated models

**Files:** `datasets/loader.py` (rewrite `_is_eligible`/`load_jsonl_records`), new `datasets/eligibility.py`.

**Design:** parse → evaluate → count. Never evaluate raw dicts.

```python
# datasets/eligibility.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

Task = Literal["text_training", "translation_training", "conversation_training",
               "asr_training", "tts_training", "evaluation", "public_release"]

_REQUIRED_SCOPE: dict[str, str] = {
    "text_training": "text_training",
    "translation_training": "text_training",
    "conversation_training": "text_training",
    "asr_training": "speech_recognition",
    "tts_training": "voice_synthesis",
    "public_release": "public_release",
}

@dataclass(frozen=True)
class EligibilityPolicy:
    task: Task
    split: str
    validated_only: bool = True            # default to the safe choice
    allow_partially_validated: bool = False
    include_machine_generated: bool = False
    require_consent: bool = True
    enforce_source_register: bool = True   # allow-list (PA-008)
    exclude_benchmark_overlap: bool = True
    allow_templates: bool = False          # tests only

def reasons_ineligible(meta, record, policy, register) -> list[str]:
    """Return every reason (empty list == eligible). `meta` is a validated DatasetMetadata."""
    r: list[str] = []
    if not policy.allow_templates and meta.source.upper().startswith(TEMPLATE_PREFIXES):
        r.append("template_record")
    if meta.validation_status is ValidationStatus.REJECTED:
        r.append("rejected")
    if meta.validation_status is ValidationStatus.MACHINE_GENERATED and not policy.include_machine_generated:
        r.append("machine_generated")
    if policy.validated_only:
        ok = meta.validation_status is ValidationStatus.VALIDATED or (
            policy.allow_partially_validated
            and meta.validation_status is ValidationStatus.PARTIALLY_VALIDATED)
        if not ok:
            r.append(f"not_validated({meta.validation_status.value})")
    if meta.licence_id in (None, "unknown") or not meta.training_permitted:
        r.append("rights_not_cleared")
    if meta.consent_withdrawn:
        r.append("consent_withdrawn")
    if policy.require_consent:
        if not meta.contributor_consent:
            r.append("no_consent")
        need = _REQUIRED_SCOPE.get(policy.task)
        if need and need not in meta.consent_scope:
            r.append(f"consent_scope_missing({need})")
    if meta.split.value != policy.split:
        r.append(f"split_mismatch({meta.split.value}!={policy.split})")
    if policy.enforce_source_register:
        row = register.get(meta.source)
        if row is None:
            r.append("source_not_in_register")                    # allow-list
        elif row.get("decision", "UNREVIEWED").upper() != "APPROVED":
            r.append("source_rights_not_approved")
    return r
```
Loader changes (`load_jsonl_records`):
1. Read with `encoding="utf-8-sig"` (PA-009b).
2. For each line: `json.loads` → if not `dict` or `metadata` not `dict` → `LoadIssue` (never raise `AttributeError`).
3. `model = model_cls(**item)` in `try/except ValidationError` → `LoadIssue`.
4. **Then** `reasons = reasons_ineligible(model.metadata, model, policy, register)`; count each reason in `skipped`.
5. Duplicate `record_id` after eligibility.
6. Benchmark overlap on **all text-bearing fields** (see PA-004 `iter_texts(model)`).
7. Public signature: keep `load_krio_text(...)` etc. as thin wrappers over a **generic** `load_records(language, kind, split, policy)` (fixes PA-021/the Krio hard-coding).
8. `translation_method == "machine_only"` is always excluded; `TranslationRecord` with `licence`… same policy.

**Tests (all fail on current code):**
- record with no `licence` key → not eligible (`rights_not_cleared`).
- `licence=proprietary`, `consent_scope=["voice_synthesis"]`, task `text_training` → `consent_scope_missing(text_training)`.
- `metadata="x"` in strict mode → `DataValidationError` with line number (not `AttributeError`).
- UTF-8-BOM file loads (BOM written with `b"\xef\xbb\xbf"`).
- source not in register → excluded; source `APPROVED` → included.

---

## PA-001 / PA-007 — Readiness from eligible, validated records via a manifest

**Files:** `datasets/loader.py::_count_real_records` (delete), `training/guard.py::check_dataset_readiness`, new `datasets/manifest.py`, CLI.

1. **Counting** = number of records returned by `load_records(..., strict=True, policy=EligibilityPolicy(task=…, split="train"))` per task/directory, **de-duplicated by `record_id` across directories** (cleaned + validated must not double-count).
2. **Skip** `quarantine/`, `splits/`, `manifests/`, files named `*.issues.jsonl` and `example_*`.
3. **Fix directory map:** text → `validated/` (once CPT/SFT inputs are decided — pick **one**: `validated/` for training; `cleaned/` is pre-review), translations, conversations, audio manifests (`audio/*.jsonl`), TTS = audio records whose consent includes `voice_synthesis` (no separate `tts/` folder needed).
4. **Manifest:** `slai build-manifest <language> --split train` writes `datasets/manifests/<language>-<split>-vN.manifest.json`:
   ```json
   {"schema_version":2,"language":"krio","split":"train",
    "files":{"datasets/krio/validated/krio_text_train.jsonl":"<sha256>"},
    "eligible_counts":{"text_training":0,"translation_training":0,"conversation_training":0,"asr_training":0,"tts_training":0},
    "skipped":{"no_consent":0},"built_at":"…","git_commit":"…"}
   ```
   `check_dataset_readiness` reads **the manifest** and re-verifies the hashes (cheap), so readiness cannot drift from what training will read; the training approval manifest (PA-010) references the dataset manifest id.
5. **Thresholds:** keep `_MIN_RECORDS` but put it in a versioned file `training/readiness_thresholds.json`; document that 50,000 is a *planning target* (see roadmap — pilot gates are smaller).
6. **Tests:** (a) 3 unvalidated/no-consent/unknown-licence/wrong-split rows + 1 machine-generated → count **0** (this is exactly the case that returned 4); (b) a quarantine `.issues.jsonl` → 0; (c) the same `record_id` in `cleaned/` and `validated/` → 1; (d) monkeypatch thresholds to 2 and assert a *passing* case turns green.

---

## PA-004 — Benchmark integrity

**Files:** `evaluation/benchmarks.py`, `datasets/schema.py` (new `BenchmarkItem`), `benchmarks/krio/held_out_eval_schema.jsonl` (template only).

1. Add `BenchmarkItem(BaseModel)`: `metadata: DatasetMetadata` (with `split=held_out`), `item_id`, `evaluation_category`, `prompt: str`, `references: list[str]` (≥1; multiple allowed for orthographic variants), `reference_notes`, `review_count`, `benchmark_version`.
2. Generic text iterator used by loader and benchmark module:
   ```python
   def iter_texts(obj: dict) -> Iterator[str]:
       for k in ("text","source_text","target_text","question","answer","transcript","prompt","reference","references"):
           v = obj.get(k)
           if isinstance(v, str) and v.strip(): yield v
           elif isinstance(v, list): yield from (x for x in v if isinstance(x, str) and x.strip())
       for t in obj.get("turns", []) or []:
           if isinstance(t, dict) and isinstance(t.get("text"), str): yield t["text"]
   ```
3. `assert_held_out_integrity` returns `True` only if: ≥ `MIN_BENCHMARK_ITEMS` (config; default 300), every item validates as `BenchmarkItem`, every item has `review_count >= 2`, and the **benchmark manifest hash matches** (`benchmarks/manifests/krio-vN.manifest.json`, built by `slai build-benchmark-manifest`). Return a structured result `{ok, count, reasons}` rather than a bare bool; keep a bool wrapper.
4. Move `TEMPLATE_PREFIXES` to `datasets/constants.py` and import it from there in loader, benchmarks and tokenizer_audit (PA-021).
5. Do not skip files by name (`example_`, `held_out_eval_schema`) — skip by *template source* only, plus an explicit `*schema*.jsonl` rule documented once.
6. **Tests:** a benchmark row with `prompt` → one fingerprint; a conversation row → its turn texts; a training row equal to a benchmark prompt (case/punctuation changed) is dropped with reason `benchmark_overlap`; editing a benchmark file after the manifest was built → `ok=False`.

---

## PA-005 — Make the registry truthful

1. **Edit now (`models/registry.py`):**
   - NLLB entry: change `african_language_support` to: *"Krio (`kri_Latn`) is NOT in the NLLB-200 language list on the official model card. Languages of the region it does list include Wolof, Hausa, Yoruba, Igbo, Twi — verify before use."* Keep it as a **translation baseline for other languages only**, or remove it. Add note: card says research model, not for production; CC-BY-NC-4.0.
   - "Tiny Aya (Aya-Expanse variants)": replace with the real Tiny Aya family (**verified on the Hugging Face model cards**): `CohereLabs/tiny-aya-base`, `CohereLabs/tiny-aya-global`, `CohereLabs/tiny-aya-earth` (Africa + West Asia), 3.35 B parameters, 70 languages, 8 k context, **licence CC-BY-NC-4.0 + Cohere acceptable-use policy (non-commercial; gated download)**. The card's African-language list excerpt shows `am, ha, ig, mg, sn, sw, wo, xh, yo, zu` — **no `kri` seen** (full list not confirmed). Keep `krio_support="unknown"`. If you want Aya Expanse 8B, add it as a separate entry with its own card facts. Never point one name at the other's repo.
   - `multilingual` flags: set from the model card text; if unsure set `None` (change type to `bool | None`).
2. **Structural fix:** move entries to `models/candidates.yaml`; `CandidateModel` loaded with Pydantic; fields `source_url` and `verified_on` **required** for `verified: true` entries; `krio_support` as an enum `{"documented","undocumented","unsupported","unknown"}` with a mandatory `evidence_url` when `documented`.
3. **CI check:** `tests/test_registry.py` fails if any entry with `krio_support="documented"` lacks `evidence_url`, or any entry lacks `source_url`; remove the `pytest.skip` at line 89.
4. **Add (metadata only, `verified_on=None` until a human checks the card):** `KhayaAI/w2v-bert-men`, `KhayaAI/w2v-bert-tem` (monolingual DONDO checkpoints exist), keep `w2v-bert-kri-men-tem-en`; note "trained primarily on read religious speech; reported in-domain WER 10.6 % (Temne), 12.9 % (Mende) — self-reported, in-domain" (from the model cards).

---

## PA-010 — Bind the training approval

**File:** `training/guard.py`

1. In `require_training_approval`, after loading the manifest, **assert**:
   ```python
   if manifest.task != task_key:            # e.g. "chat_llm_sft"; add a `task_key` parameter
       raise RuntimeError("manifest task does not match requested task")
   selected = get_recommended(...)          # reads models/selected_base_model.json
   if selected is None or selected.hf_repo != manifest.base_model_id:
       raise RuntimeError("manifest base_model_id != selected base model")
   ```
2. Require `manifest.split_manifest_id` to exist as a file and `manifest.dataset_manifest_id` (new) to match the dataset manifest from PA-001; the config's data paths must be ⊆ the manifest's hashed files.
3. Extend the dry-run plan with: resolved config paths, eligible counts per task, tokenizer id, `torch.cuda.is_available()` and device names **only if torch is installed** (`try: import torch`), estimated GPU-hours field from the manifest, and `"downloads": 0, "writes": 0`.
4. `ModelLicenceReview.reviewed_on` → `date`; require `commercial_permitted` to be `True` if `intended_use == "commercial"` (new field).
5. Document in `documentation/SYSTEM_ARCHITECTURE.md`: *"The guard prevents accidental spend and unreviewed data; it is not an access-control system — anyone with repo write access can author a manifest."*
6. **Tests:** manifest for a different model → raises; manifest referencing a file not covered by the dataset manifest → raises; dry-run output contains `eligible_counts`.

---

## PA-012 / PA-013 — API hardening (still no model, all model endpoints stay 503)

**Files:** `api/app.py`, `api/models.py`, `config.py`

1. **Language validation once:**
   ```python
   def _require_language(code: str, *, allow_english: bool = True) -> str:
       c = code.lower()
       if c not in LANGUAGE_REGISTRY or (c == "english" and not allow_english):
           raise HTTPException(422, {"error": "unsupported_language", ...})
       return c
   ```
   Use in `/generate` (`allow_english=False`), `/translate`, `/synthesize`. In `/translate` require **exactly one side English**: `if (src == "english") == (tgt == "english") → 422 "translate requires English on exactly one side"`. Delete lines 210-215 duplicates.
2. **Models:** type `GenerateRequest.language`, `TranslateRequest.*` with the `Language` enum (or a shared `LanguageCode = Annotated[str, AfterValidator(...)]`).
3. **Readiness from state, not a dict:** add `sierraleone_ai/serving/state.py` with `ModelState` (loaded model ids per task). `/health/ready` is 200 only if `ModelState.is_ready()`; `LANGUAGE_REGISTRY["status"]` becomes *documentation*, not the gate.
4. **Uploads:**
   - Add `audio/mp4, audio/x-m4a, audio/aac, audio/3gpp, audio/amr` and magic prefixes: `b"\x00\x00\x00" + ftyp at offset 4`, `#!AMR`, `\xff\xf1`/`\xff\xf9` (ADTS AAC). **[verify]** the exact byte signatures against a reference before shipping.
   - WAV check: `payload[:4] == b"RIFF" and payload[8:12] == b"WAVE"`.
   - Stop writing the upload to disk while the endpoint is a stub; when a model exists, stream to a `tempfile.SpooledTemporaryFile` and delete in `finally`.
   - **Size cap at the edge:** reject early from the `Content-Length` header with a tiny ASGI middleware, and set `client_max_body_size` at the reverse proxy (nginx/Caddy). **[verify]** Starlette's current multipart limits in your installed version — if it spools before the handler, the header/proxy check is the real control.
5. **CORS:** `SLAI_CORS_ORIGINS` documented as JSON (`["https://example.org"]`); default empty means *no CORS* (remove the hard-coded `localhost:3000` fallback or set it only when `SLAI_API_DEBUG=true`). Add tests with `TestClient` + `Origin` header.
6. **Before any real model:** API keys (`fastapi.security.APIKeyHeader`) or JWT, per-key rate limiting (reverse proxy or `slowapi` **[verify maintenance]**), request-ID middleware, structured logs without payloads. Not needed while everything is 503.
7. **Docs:** update `API_DOCUMENTATION.md` (health endpoints, 413/415/422, correct run commands, no `--host 0.0.0.0 --workers 4` "production").

---

## PA-014 — Configuration

1. `.env.example`: remove `SLAI_DATA_DIR`, `SLAI_MODELS_DIR`, `SLAI_LOGS_DIR` lines (defaults are anchored); comment them as "override only with an **absolute** path". `SLAI_API_HOST=127.0.0.1`. Add `SLAI_CORS_ORIGINS=[]` with a JSON note, `SLAI_MAX_AUDIO_UPLOAD_BYTES=10485760`.
2. `config.py`: validate that overridden paths are absolute (`@field_validator(...)`) or resolve them against `project_root`; set `env_file=_DEFAULT_ROOT / ".env"` so the `.env` is found regardless of CWD.
3. Delete unused settings (`api_host/port/debug`, `logs_dir`) **or** wire them: add `slai serve` (`uvicorn.run(app, host=settings.api_host, port=settings.api_port)`) and configure loguru to write to `logs_dir`.
4. Use `settings.hf_token` in `load_hf_tokenizer_only`:
   ```python
   tok = settings.hf_token.get_secret_value() or None
   AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=False, token=tok)
   ```
   **[verify]** the `token=` kwarg name for your `transformers` version.
5. Tests: `.env` in a temp CWD is found; relative `SLAI_DATA_DIR` is rejected/resolved; `repr(settings)` contains no token.

---

## PA-015 — Reproducible environments and CI

1. **Lockfile:** `uv lock` (project already built with uv's interpreter) → commit `uv.lock`; CI uses `uv sync --frozen --extra dev --extra eval`. Delete `constraints.txt` or generate it with `uv export --no-hashes > constraints.txt` and **use it** (`pip install -c constraints.txt`).
2. **CI** (`.github/workflows/ci.yml`):
   ```yaml
   permissions: { contents: read }
   strategy:
     matrix:
       os: [ubuntu-latest, windows-latest]
       python-version: ["3.10", "3.12", "3.13"]
   steps:
     - uses: actions/checkout@v4          # consider pinning to a commit SHA
     - uses: astral-sh/setup-uv@v5        # [verify current major]
     - run: uv sync --frozen --extra dev --extra eval
     - run: uv run ruff check . && uv run ruff format --check .
     - run: uv run mypy src
     - run: uv run pytest --cov=sierraleone_ai --cov-report=term-missing --cov-fail-under=<measure first, then set>
   ```
   Add a job that installs `.[training]`/`.[audio]` on Linux **with `--dry-run`/import-only** (`python -c "import peft, trl"`) — optional, no GPU.
3. Broaden the "no Krio in tests" step: scan `tests/ src/ api/` for a *file* `tests/krio_strings_denylist.txt` instead of a grep with `bo,`; fail only on whole-word matches (`grep -rwnE`).
4. `.pre-commit-config.yaml`: bump ruff hook to the version in `uv.lock` (or run `pre-commit autoupdate`); add `detect-secrets` or `gitleaks`.
5. Add Python 3.14/3.15 as **allowed-failure** experimental jobs (you develop on 3.15.2).

---

## PA-016 — CLI that tells the truth

- `slai validate-file PATH --kind {text,translation,conversation,audio,benchmark} [--task …] [--split …]`: call the same `load_records(..., strict=False)` and print: total lines, valid, **per-reason skip counts**, first 10 issues with line numbers, PII scan results; exit code 1 on any issue.
- `slai status`: compute every row — package version, `import` checks for optional extras, `git rev-parse --short HEAD`, dataset manifest age + eligible counts, benchmark `ok/reasons`, registry `selected_base_model.json` presence, API readiness (`ModelState`), GPU presence (only if torch importable). States: `implemented | scaffolded | not configured | not ready | operational` — "implemented" must mean *code exists and its self-check passed*, never "thresholds met".

---

## PA-017 — Documentation corrections (checklist)

- [ ] `DATASET_FORMAT.md`: regenerate every example from the pydantic models via a small script (`scripts/gen_schema_examples.py`) using `TEST_FIXTURE` strings, include `reviews`, `consent_scope`, `licence_id`, provenance fields, `schema_version`, `contributor_id`; add a doctest-style test that parses each JSON block in the doc.
- [ ] `CLAUDE.md`: fix test counts (use `pytest --collect-only -q | tail -1`), `_model_not_ready()`, rewrite "Adding a New Language" (edit `LANGUAGE_REGISTRY` + `Language` enum, create dirs, benchmark dir, add a loader entry), endpoints = 8, and mark Rule 2's examples **"unverified – pending native review"**.
- [ ] `API_DOCUMENTATION.md`: see PA-012 item 7.
- [ ] `DEPENDENCIES_AND_LICENCES.md`: regenerate with `pip-licenses --format=markdown` (a CI step that fails on drift), remove `jsonschema`, fix cross-reference.
- [ ] `SYSTEM_ARCHITECTURE.md`: registry is per-task; add threat model (PA-010).
- [ ] `docs/NATIVE_REVIEW_QUEUE.md`: delete rows for strings no longer present; add rows for the remaining samples in `example_*.jsonl` and docs.
- [ ] `IMPLEMENTATION_CHANGELOG.md`: add a mapping table old→new IDs; correct the five inaccuracies listed in the audit.
- [ ] `pyproject.toml [project.urls]` → `https://github.com/Morie55/SierraLeone-AI-Foundation` (or move the repo to the org and update `origin`).
- [ ] README: add `speech/` to the tree; show both `copy` and `cp`; link the two new docs; replace "continued pre-training and instruction fine-tuning" with the baseline-first plan (PA-024).

---

## PA-018 — Metrics

1. Reporting path: `sacrebleu` (`BLEU` and `CHRF(word_order=2)` for chrF++) returning the **signature string** with every score; wrap in `evaluation/reporting.py`.
2. Keep `bleu_score`/`chrf_score` only as documented toy helpers in `evaluation/_toy_metrics.py`; rename `chrf_score` → `char_ngram_f_toy` and fix docstrings (no "chrF++" claim).
3. `_edit_distance`: two-row DP (O(min(n,m)) memory).
4. Tests: compare `reporting.chrf` to a hand-computed sacrebleu value for a fixed fixture; assert the signature is present.

---

## Low-severity fixes (PA-019 … PA-025)

| ID | Fix |
|---|---|
| PA-019 | `_WHITESPACE_PATTERN = re.compile(r"[ \t\r\f\v]+")`, normalise `\r\n`→`\n` first; run the overlong check **after** whitespace normalisation; keep marker lists "needs review" and remove assertions from CLAUDE.md Rule 2; rename `comot` handling to a *spelling-variant note*, not an overlap flag. |
| PA-020 | Wrap `json.loads` and report `LoadIssue`-style errors; `require_validated` default `VALIDATED` only (param to widen); record `tokenizer_revision` (`tok.init_kwargs`/repo commit **[verify]**) and library versions in the report. |
| PA-021 | Remove `ReadinessItem` aliases and update tests; `datasets/constants.py` for prefixes; make `LoadReport` a plain dataclass without `__eq__`/list-compat (tests use `.records`); `stats.py` header "Records"; stream JSONL (generator) and cache register/fingerprints per process with `functools.lru_cache` + explicit clear in tests. |
| PA-022 | Atomic write (`tmp = path.with_suffix(".tmp"); tmp.write_text(...); os.replace(tmp, path)`); optional `weight_field="duration_seconds"` to balance splits by hours; log final ratios. |
| PA-023 | `check_tts_recording_quality(..., dataset_root=…)` calls `verify_audio_file`; compare environment with `.casefold()`; enforce `ASRTrainingConfig` duration limits in a `validate_asr_record()`; default ASR base model chosen from the registry decision file, not hard-coded. |
| PA-024 | Replace the default recipe with an **experiment ladder** (below); LR chosen per method; `target_modules="all-linear"` when supported **[verify PEFT version]** else discover by architecture; add `modules_to_save` guidance only when the vocabulary is extended; `num_train_epochs` + early stopping from a pilot; make the pipeline stage names `baseline_eval → sft_lora → (optional) cpt`. |
| PA-025 | Remove `filterwarnings` ignores (fix the warnings or pin narrowly); delete `pytest.skip`; add the regression tests named above; add a Windows CI job; add property-style tests for `deterministic_split` (hypothesis) **[optional dependency]**. |

### Experiment ladder (PA-024) — what the pipeline should encode
1. `baseline_eval`: zero/few-shot each candidate on the locked benchmark (no training).
2. `sft_lora`: small LoRA SFT on native-reviewed pairs; compare against (1) with the same prompts/metrics.
3. `cpt` (optional): only if (a) corpus ≥ a threshold justified by (1)–(2) and (b) tokenizer audit shows a language-modelling gap.
Each stage writes a run record (git commit, lock hash, dataset manifest id, model revision, seed, hardware, metrics) to `runs/<id>.json` (committed; no data inside).

---

## Data migration & API change summary

**Database migrations:** none (no database). **File-format migration:** JSONL `schema_version` 1→2 via `scripts/migrate_schema_v1_to_v2.py` (PA-011).
**API changes:** `/translate` requires exactly one English side (422 otherwise); language fields become enums (422 bodies standardised); new `/health/ready` semantics (state-based); additional accepted audio MIME types; documented 413/415/422; CORS default becomes "none".
**CLI changes:** `validate-file --kind`, `status` computed, new `build-manifest`, `build-benchmark-manifest`, optional `serve`.
**Config changes:** `.env.example` trimmed; `env_file` anchored; `SLAI_CORS_ORIGINS` documented.

---

## Definition of done (all must be true)
- Executing the audit's failing cases against the new code now **passes** (rows → `not eligible`; readiness count `0`; benchmark `prompt`/`turns` fingerprinted; BOM loads; missing-licence record rejected).
- `pytest`, `ruff check`, `ruff format --check`, `mypy src` green on Ubuntu **and** Windows, Python 3.10/3.12/3.13.
- With only template files present: readiness NOT READY for every task; `require_training_approval` raises even with `SLAI_ALLOW_TRAINING=true` and a syntactically valid manifest for a non-selected model.
- `git ls-files datasets benchmarks` contains only allowed templates/manifests.
- No new Krio text, no model downloads, no network calls in tests.
