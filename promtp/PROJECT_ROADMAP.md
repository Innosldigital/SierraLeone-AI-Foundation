# PROJECT_ROADMAP.md — Sierra Leone AI Foundation

*Companion to `PLATFORM_AUDIT.md` and `IMPLEMENTATION_FIXES.md`. Date: 2026-11-11.*
Evidence tags: **[W]** verified on an official/primary page during this audit · **[S]** from a secondary source or another reviewer's report, **not independently verified** · **[U]** my judgement/recommendation — validate before relying on it.

---

## 1. Honest starting point

**What exists:** a well-structured Python package that encodes *data governance* (consent, rights, validation, split, benchmark, approval gates), plus a FastAPI stub, CI and docs. **What does not exist:** any Krio data, any benchmark, any model, any frontend, database, authentication or deployment. The README says this correctly.

**What is still wrong:** the gates meant to protect future data have holes (`PA-001…004`) and the model registry contains a false claim (`PA-005`). Fix those first; they are cheap and they matter most when data starts arriving.

**Your goal ("a reliable, secure, scalable, production-ready platform")** is reachable, but the order matters: *trustworthy data and evaluation → measured baselines → small adaptation → serving → scale*. A bigger model or more code does not substitute for any of those steps.

---

## 2. The landscape you are entering (researched)

| Fact | Status | Why it changes your plan |
|---|---|---|
| **DONDO** (Khaya AI, arXiv 2607.21540, July 2026): 21 monolingual + 5 multilingual **Apache-2.0** w2v-BERT 2.0 CTC ASR models; includes a Sierra Leone multilingual model `kri-men-tem-en` and monolingual Mende/Temne checkpoints. Trained **primarily on read religious speech**; self-reported in-domain WER 10.6 % (Temne), 12.9 % (Mende) | **[W]** (paper listing + model cards for `w2v-bert-men`, `w2v-bert-tem`; the `kri-men-tem-en` card itself not opened) | You do **not** need to train ASR from scratch. Evaluate it on farm/phone speech; fine-tune with a small consented set. |
| Khaya's API/SDK lists ASR for `kri`, `men`, `tem`; its translation pairs list does **not** include Krio | **[W]** (PyPI `khaya` 0.2.0 page) | A hosted ASR baseline exists; translation does not. |
| **NLLB-200 does not list Krio** (`kri_Latn` absent); research model, CC-BY-NC-4.0 | **[W]** official model card | Remove it as a Krio MT baseline (registry fix PA-005). |
| **Google Translate has supported Krio since May 2022** (zero-shot, monolingual) | **[W]** (earlier audit round: Google blog/TechCrunch) | A free baseline you must beat on *your* benchmark; its output is `machine_generated` data only. |
| **Tiny Aya** (Cohere Labs, Feb 2026): 3.35 B, 70 languages, 8 k context, `base/global/earth/fire/water`; **CC-BY-NC-4.0 + Cohere AUP, gated**; African list excerpt shows `am ha ig mg sn sw wo xh yo zu` (no `kri` seen) | **[W]** model cards | Good *research* candidate; **non-commercial licence** blocks commercial use. |
| **Qwen3-4B-Base**: Apache-2.0, 36 T tokens, 119 languages | **[W]** (model-card mirrors; open the HF card yourself to confirm) | Commercially usable candidate; Krio capability unproven. |
| **AfriqueLLM** (ACL 2026): continued pre-training on ~26 B tokens across 20 African languages (Llama-3.1-8B, Gemma-3, Qwen3) | **[W]** (earlier round: arXiv 2601.06395) | Your ≈1 M-token Krio corpus is ~4 orders of magnitude smaller → CPT-first is the wrong default. |
| Two **Sierra Leone Krio projects** were LINGUA Africa awardees in 2026: *Krio Voices* (Christex Foundation), *SAM App* (Institute for Development) | **[W]** (earlier round: Microsoft Research page) | Same country, same language: collaborate, don't duplicate. |
| **Kuma (DSTI, Sierra Leone government)** runs a language-AI programme (translation/TTS/STT for Krio, Temne, Mende, Limba) and a research page; **SierraLeoneNLP** exists as a Hugging Face org | **[W]** pages exist; details of data/models/licences are **[S]** from another reviewer's summary | Contact first; ask what data/benchmarks/checkpoints are shareable and on what terms. |
| Umeå University *Krio Corpus Project* (written Krio, since 1981); `Jaward/Krio-Corpus` (tiny, cites published books) | **[W]/[S]** | Possible text sources; **rights unresolved** — keep `UNREVIEWED` in the register. |

**Licence matrix for candidate use in a possibly commercial product (CAFP):** Qwen3 (Apache-2.0) ✔ · DONDO (Apache-2.0) ✔ · Gemma 3 (Gemma Terms — read them) ⚠ · Llama 3.x (community licence) ⚠ · **Tiny Aya, Aya-23, NLLB (CC-BY-NC) ✘ for commercial use**. Decide *now* whether CAFP is commercial; it prunes the candidate list.

---

## 3. Strategy (what to build, in one paragraph)

Build **Krio language technology for agriculture**, as a *system*, not a giant model: (1) a small, native-reviewed, consented corpus and a locked benchmark; (2) measured baselines using existing models (LLM prompting, Google Translate, DONDO ASR); (3) the smallest adaptation that measurably beats the baseline (retrieval over verified agronomy content + a LoRA fine-tune); (4) a separate inference service behind your API; (5) integrate into the CAFP Farmer App through that API. Facts about farming come from CAFP's verified knowledge base (retrieval), **not** from the language model; the model handles Krio language and dialogue.

---

## 4. Recommended target architecture

```
Contributors/Reviewers ──► Collection & review tools ──► Consent store (restricted)
                                   │
                                   ▼
        Private data store (HF private datasets or DVC + S3/R2)  ◄── manifests + hashes in git
                                   │  strict loader + ONE eligibility policy
                                   ▼
        Dataset manifests ─► Readiness/approval gate ─► Experiment runner (baseline → LoRA → [CPT])
                                   │                          │
                 Locked native-reviewed benchmark ◄───────────┘ (scores + signatures)
                                   ▼
                         Model registry (data file) + model cards
                                   ▼
        Inference service (vLLM / llama.cpp)  ◄─ separate process/host, no web code
                                   ▲
        API gateway (FastAPI: auth, rate limit, logging)  ◄─ CAFP Farmer App backend ◄─ farmers
                                   ▲
                         Retrieval layer over verified agronomy content (CAFP)
```

| Layer | Recommended technology | Why / verification |
|---|---|---|
| Schemas & validation | **Keep** Pydantic v2 + strict loader (after PA-001…003) | Already good foundation. |
| Packaging/env | `uv` + lockfile, Python 3.10–3.13 in CI (+3.14/3.15 allowed-fail) | You already use uv's interpreter. [U] |
| Data storage | Pilot: encrypted local drive + manifests. Next: **private Hugging Face dataset repos** or **DVC + S3-compatible bucket** | Keeps data out of git (PA-006). Check current storage limits/pricing before committing. [U] |
| Consent records | Restricted store separate from data (encrypted spreadsheet for the pilot → small PostgreSQL later) | PII must not live in JSONL; no DB needed before ~hundreds of contributors. [U] |
| Review/annotation | Pilot: structured form + spreadsheet; then **Label Studio** or **Argilla** (open source) | Native reviewers need a workflow that records `reviewer_id` and verdict. [U — evaluate both] |
| Training | `transformers` + `peft` + `trl` on **Linux** GPU (WSL2/cloud); `bitsandbytes` only if verified on the chosen GPU/OS | `bitsandbytes` is unreliable on native Windows (your own docs say so); T4 lacks fast bf16. [U] |
| Experiment tracking | Plain JSON run records in git (no data) → optionally MLflow/W&B later | Cheap, reproducible. |
| Evaluation | `sacrebleu` (BLEU, chrF++), WER/CER library, **native-speaker panel**; AfroBench-style tasks for sanity | Automatic metrics are supplementary. |
| Serving | Small models: **llama.cpp/GGUF** (CPU/edge; Tiny Aya GGUF exists for reference) or **vLLM** (GPU, OpenAI-compatible endpoint) | Both appeared in current model-card usage docs. Choose after the model is chosen. [W/U] |
| API gateway | **Keep FastAPI** for auth (API keys/JWT), rate limiting (proxy or library), request IDs, logging; proxies to the inference service | Don't load multi-GB weights into the web process. |
| Observability | Structured JSON logs (no payloads), Prometheus-style metrics or OpenTelemetry | Standard. [U] |
| Deployment | Docker image(s) + CI/CD; one small VM or a GPU VM only when a model is approved; offline-friendly variants for low connectivity | Connectivity in rural Sierra Leone is a product constraint to validate with CAFP users. [U] |

### Missing platform pieces to design later
Contributor-facing collection app (text + voice, consent capture in Krio/English); reviewer UI; admin dashboard for manifests/benchmarks; **user-facing client** (the CAFP app, WhatsApp/SMS/USSD options — validate with users); abuse/safety handling for generated advice; data-retention and revocation tooling.

---

## 5. Phased plan with exit criteria

Durations are planning estimates for a small team and depend on partner access and reviewers. Each phase ends with a **go/no-go gate**; do not start spending (GPU, annotators at scale) before the gate that unlocks it.

### Phase 0 — Make the repo trustworthy (1–2 weeks; agent + you review)
- Apply `IMPLEMENTATION_FIXES.md` in the stated order; run `pytest/ruff/mypy` on Linux **and** Windows.
- Put real data **outside** git (PA-006) before any data exists.
- **Exit:** the audit's failing cases pass (§Definition of done in the fixes file); CI green on 2 OSes; no template/ineligible row can turn a readiness item green.

### Phase 1 — People, partners, governance (start now, parallel)
- Write/email Kuma (DSTI), SierraLeoneNLP maintainers, Christex Foundation (*Krio Voices*), Institute for Development (*SAM App*), Umeå Krio corpus team. Ask for: data/benchmark/checkpoint availability, **licence terms in writing**, current status, joint orthography work.
- Appoint a small **language review group**: 1 Krio linguist, 3–5 native reviewers from different regions/ages, 1 agricultural extension specialist, 1 data steward. Pay or credit them.
- Decide and document: orthography policy (do **not** force `ɛ/ɔ` vs plain `e/o` prematurely — support both, record the profile), dialect/region labels, code-switching handling, annotation guidelines.
- Governance: who owns the data, permitted uses (text / ASR / TTS / public release / commercial), contributor revocation process (`CONSENT_AND_REVOCATION.md`), community-benefit and data-sovereignty principles (e.g. CARE-style). Legal review before any public release or commercial training.
- **Exit:** signed partner contacts, consent templates (separate ASR and TTS opt-ins), annotation guideline v1, first reviewer cohort ready.

### Phase 2 — Pilot corpus + locked benchmark (≈ weeks 2–8)
Planning targets (not claims of sufficiency): 2–5 k naturally written Krio sentences from **permitted** sources; 500–1 k English↔Krio pairs (human, dual-reviewed); 100–300 multi-turn agricultural conversations **verified by an agronomist** (a fluent speaker cannot validate agronomy); a **300–500 item held-out benchmark** authored/checked by ≥2 native reviewers and never used for training; a speech pilot of 5–10 consented speakers (30–60 min each) to test the collection and transcription workflow.
- Use `schema_version 2` records, `reviews`, `consent_scope`, source-register `APPROVED` only.
- **Exit:** ≥ 300 benchmark items with `review_count ≥ 2`, benchmark manifest + hash; pilot corpus passes strict load with 0 issues; inter-reviewer agreement measured and reported.

### Phase 3 — Baselines, no training (≈ weeks 6–10)
1. **Tokenizer audit** (tokenizer files only) on 3–5 candidates (Qwen3-4B, Gemma-3 4B, Llama-3.2-3B, plus Tiny Aya for research comparison). Report fertility, `ɛ/ɔ` handling, length overhead vs English; treat as *one* criterion.
2. **Zero/few-shot benchmark** of each candidate and of hosted general-purpose LLMs on the same prompts; **Google Translate Krio** as MT baseline; **DONDO ASR** (`kri-men-tem-en` and `men/tem`) on your speech pilot (WER/CER + native review of errors, include spontaneous and code-switched speech).
3. Native-speaker scoring of outputs (`KrioFluencyEvaluation`), ≥2 raters.
- **Exit / decision gate:** a written decision record: *is the gap between baseline and "good enough for agriculture Q&A" large enough to justify fine-tuning, and which base model (licence-compatible) is selected?* Only then create `models/selected_base_model.json`.

### Phase 4 — Smallest useful adaptation (≈ weeks 10–16)
- **Retrieval first:** index CAFP's verified agronomy content; the LLM answers in Krio *from retrieved text*. Measure hallucination and factual accuracy with an agronomist.
- **LoRA SFT pilot** on native-reviewed instruction/translation pairs (thousands, not millions). Compare against the **unmodified** baseline on the locked benchmark, same prompts, same metrics, ≥2 seeds.
- **CPT only if** the baseline shows a clear language-modelling gap *and* you have orders of magnitude more clean Krio text; run as a separate controlled experiment and compare all variants.
- Use QLoRA only after verifying GPU/CUDA/`bitsandbytes`; choose LR per method (pilot; `2e-5` is not a LoRA default); discover `target_modules` from the architecture.
- **Exit:** a model card with benchmark results, known failure modes (word order, code-switching, dialect), licence compatibility, and an approved training manifest bound to the selected model (PA-010).

### Phase 5 — Speech (parallel track after Phase 3)
- **ASR:** start from DONDO; adapt only with enough speaker-diverse, consented, dual-verified audio; split by speaker; locked speaker-disjoint test set; report WER/CER **and** error review.
- **TTS (last):** single-speaker, studio-quality, explicitly consented `voice_synthesis`; pronunciation test suite; native listening tests; check model licence (MMS-TTS is CC-BY-NC per registry — **[verify]**). Do not promise a Krio voice until native listeners accept it.

### Phase 6 — Serving MVP + CAFP integration (≈ weeks 14–22)
- Stand up the **inference service** separately; the FastAPI gateway authenticates, rate-limits and logs (no payload logging); `/health/ready` reflects real model state.
- Integrate **text first** via the CAFP backend (never embed credentials in the mobile app); voice later.
- Safety: refusal/"I don't know" behaviour for out-of-scope agronomy; show provenance of retrieved sources; human-in-the-loop for high-stakes advice (pesticides, livestock disease).
- **Exit:** load test, security review (auth, upload limits, PII, secrets), rollback plan, monitoring dashboard, user pilot with farmers/extension officers and feedback loop.

### Phase 7 — Scale (after Phase 6 is stable)
Add Mende/Temne using the same pipeline (rename Krio-specific fields first); expand corpora and reviewers; scheduled benchmark regressions on every model/data change; public dataset release only for `APPROVED` rights and `public_release` consent; consider a bigger base model only if the benchmark justifies it.

---

## 6. Priorities

| Priority | Item |
|---|---|
| **P0 (now)** | PA-001…006 (gates, benchmark, registry truth, data-out-of-git); partner outreach; governance/consent templates; recruit reviewers |
| **P1 (next 4–8 wks)** | PA-007…018; pilot corpus + benchmark; baselines; CI on Windows+Linux; lockfile |
| **P2 (later)** | PA-019…025; annotation tooling; serving; CAFP integration; speech adaptation; TTS |

**Incomplete features (by design today):** training loops (stubs), ASR/TTS pipelines, inference, auth, collection tooling, annotation workflow, evaluation aggregation (only forms exist), dataset release tooling.

---

## 7. Cross-cutting quality goals

**Security:** secrets only via env/secret manager (`SecretStr`); CI secret scan (+ pre-commit `gitleaks`/`detect-secrets`); API keys/JWT and rate limits before any real endpoint; uploads validated at the proxy and handler; no raw text/audio in logs; encrypted storage for consent records; retention + revocation process; threat model written down (the training guard is an accident guard, not access control).
**Scalability:** keep data generation reproducible from raw + versioned preprocessing; stream JSONL instead of loading whole files; Parquet/Arrow only in the private store; separate inference service so the web tier scales independently; cache the source register/fingerprints.
**Performance:** quantised small models for edge/CPU (GGUF); batching only on GPU servers; measure latency/throughput in Phase 6, don't guess.
**Maintainability:** one eligibility policy; one language registry; registry/thresholds/config as data files with schemas; every external claim carries `source_url` + `verified_on`; docs generated from code where possible; a changelog keyed by stable IDs.

---

## 8. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Rights/consent problems with scraped or book-derived text | High | Allow-list register, legal review, prefer community-authored data |
| Benchmark contamination (e.g. sharing the repo/zips with AI tools) | Medium-High | Benchmark outside git; deny-by-default archive; manifest hashes |
| Too little data for CPT; wasted GPU spend | High | Baseline-first ladder; decision gates; no GPU before Phase 4 |
| Reviewer disagreement on orthography/dialect | High | Multiple references, orthography profile, adjudication rule |
| Duplicating Kuma/SierraLeoneNLP/Christex work | Medium | Outreach in Phase 1; shared schema/benchmark |
| Non-commercial model licences block CAFP | Medium | Decide commercial intent now; prefer Apache-2.0/permissive candidates |
| Hallucinated agronomy advice | Medium-High | Retrieval over verified content; agronomist review; refusals |
| Phone-audio formats/noise break ASR | Medium | Accept m4a/aac/amr; test on real field audio |
| Single-developer bus factor | Medium | Docs generated from code, CI, runbooks |

---

## 9. Decisions only you can make
1. Is the CAFP product commercial (affects model licences and data licences)?
2. Who owns and governs the data (CAFP, a foundation, a community body)? What may be released publicly?
3. Orthography stance (support both `ɛ/ɔ` and `e/o`?) and who adjudicates.
4. Budget ceiling for reviewers, collection and GPU time, and which phase gates release money.
5. Whether to partner with Kuma/SierraLeoneNLP/Christex on a shared benchmark and consent template.
6. Where real data will live (private HF org, DVC bucket, institutional storage).

---

## 10. What I could not verify (treat as open)
- Whether `KhayaAI/w2v-bert-kri-men-tem-en`'s own model card matches the paper's description (only the paper listing and the `men`/`tem` cards were opened).
- Whether any Tiny Aya variant includes Krio (the language-list excerpt I saw does not show `kri`; the full list was not confirmed).
- Gemma 3 and Llama 3.2 licence terms and language coverage (not re-checked this round).
- Kuma/SierraLeoneNLP's actual data availability, licences and current status.
- Whether Mozilla Common Voice has a Krio locale (a second reviewer reported none; not checked).
- Starlette's multipart spooling behaviour on your installed version (PA-012).
- Whether `pytest`, `ruff`, `mypy` pass on your latest commit (not executed here).
