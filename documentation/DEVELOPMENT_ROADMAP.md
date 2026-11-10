# Development Roadmap

## Phase 1 – Foundation ✅ (Current)

- [x] Project structure created
- [x] Python package scaffolded
- [x] Dataset schemas defined (Pydantic)
- [x] Preprocessing pipeline (Krio normaliser)
- [x] Evaluation framework (automated + native speaker forms)
- [x] API stubs (FastAPI)
- [x] Documentation suite
- [x] Unit tests

**Blocking:** All subsequent phases require completion of Phase 2 data collection.

---

## Phase 2 – Krio Data Foundation 🔴

- [ ] Collect 50,000+ Krio text sentences
- [ ] Collect 10,000+ English–Krio translation pairs (human-verified)
- [ ] Collect 1,000+ Krio conversation dialogues
- [ ] Record Krio audio (target: 100+ hours for ASR)
- [ ] Build held-out evaluation benchmark (1,000 sentences, validated)
- [ ] Establish native speaker review panel (minimum 3 reviewers)
- [ ] Complete validation workflow for all collected data

**Estimated timeline:** 3–6 months (depends on community engagement)

---

## Phase 3 – Base Model Selection 🟡

- [ ] Measure tokenisation efficiency for each candidate model on Krio text
- [ ] Benchmark inference performance on available hardware
- [ ] Evaluate existing Krio/African language support
- [ ] Confirm licence compatibility
- [ ] Select and document the chosen base model
- [ ] Obtain approval before downloading model weights

**Blocking:** Requires Phase 2 Krio text sample (at minimum 1,000 sentences)

---

## Phase 4 – Krio Language Model 🔴

- [ ] Set up training infrastructure (GPU access confirmed)
- [ ] Run continued pre-training on Krio text corpus
- [ ] Run instruction fine-tuning on conversations + QA pairs
- [ ] Run LoRA experiments for efficiency
- [ ] Checkpoint and version all training runs
- [ ] Initial automated evaluation

**Blocking:** Requires Phase 2 (data) + Phase 3 (base model) + GPU hardware

---

## Phase 5 – Native Speaker Evaluation 🔴

- [ ] Recruit minimum 3 independent native Krio speaker evaluators
- [ ] Run structured evaluation (200+ outputs per round)
- [ ] Document all failure modes
- [ ] Publish evaluation report
- [ ] Iterate on model based on feedback

**Blocking:** Requires Phase 4 (trained model)

---

## Phase 6 – Speech Recognition 🔴

- [ ] Collect and validate 100+ hours of Krio audio
- [ ] Adapt Whisper or similar open ASR model for Krio
- [ ] Evaluate CER and WER on held-out audio test set
- [ ] Native speaker review of transcription quality

---

## Phase 7 – Text-to-Speech 🔴

- [ ] Collect high-quality studio recordings (10+ hours minimum)
- [ ] Fine-tune open TTS model (e.g., Coqui XTTS, Kokoro)
- [ ] Native speaker evaluation of naturalness

---

## Phase 8 – Multilingual Expansion 🔵

- [ ] Add Mende language data
- [ ] Add Temne language data
- [ ] Add Limba, Loko, Kono, Kissi data
- [ ] Develop multilingual model

---

## Phase 9 – Production API 🔵

- [ ] Containerise API (Docker)
- [ ] Set up monitoring and logging
- [ ] Rate limiting and authentication
- [ ] Deploy to production infrastructure

---

## Resources Required

| Resource | Phase | Status |
|---|---|---|
| Native Krio speaker community | Phase 2+ | 🔴 Not yet organised |
| GPU hardware (≥16GB VRAM) | Phase 4+ | 🔴 Not yet available |
| Native speaker evaluation panel | Phase 5+ | 🔴 Not yet recruited |
| Budget for data collection | Phase 2+ | 🟡 Needs confirmation |
| Studio recording facility | Phase 6–7 | 🔴 Not yet arranged |
