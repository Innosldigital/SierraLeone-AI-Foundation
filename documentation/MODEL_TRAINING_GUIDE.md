# Model Training Guide

> ⚠️ **No training should begin until:**
> 1. The dataset is reviewed and approved (meeting thresholds in `training/guard.py`).
> 2. A base model has been selected and recorded in `models/selected_base_model.json`.
> 3. Hardware resources are confirmed and a signed `training/approvals/<run_id>.json` manifest is created.
> 4. `SLAI_ALLOW_TRAINING=true` is set in `.env`.

---

## Strategy: The Experiment Ladder (PA-024)

We are NOT training a large language model from randomly initialised weights, nor should we run expensive continued pre-training (CPT) first on a low-resource corpus (~1M tokens).

The platform follows a strict **baseline-first experiment ladder**:

```
[ Stage 1: baseline_eval ]
       │
       ▼
[ Stage 2: sft_lora ]
       │
       ▼
[ Stage 3: cpt (optional) ]
```

### Stage 1: `baseline_eval` (No training, zero GPU training spend)
- Zero-shot and few-shot evaluation of candidate models on the locked held-out benchmark (`benchmarks/krio/`).
- Establishes empirical ground-truth performance before any adaptation.
- Produces baseline chrF++ and BLEU metrics.

### Stage 2: `sft_lora` (Instruction Fine-Tuning)
- Parameter-efficient fine-tuning (PEFT) using LoRA or 4-bit QLoRA on verified Krio instruction/translation pairs (`datasets/krio/validated/`).
- Directly compares against Stage 1 baselines using identical prompts, metrics, and held-out benchmark items.
- Recommended LoRA learning rate: `1e-4` to `2e-4` (QLoRA) rather than full fine-tune rates (`2e-5`).
- Target modules: `"all-linear"` (supported in recent PEFT) or explicit projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
- `modules_to_save`: leave empty unless the tokenizer vocabulary was explicitly extended (e.g. `["embed_tokens", "lm_head"]`).

### Stage 3: `cpt` (Continued Pre-Training — Optional)
- Undertaken **only if**:
  1. Corpus size justified by Stages 1 & 2 results (requires sufficient scale of validated text).
  2. Tokenizer audit demonstrates a clear language-modelling gap in base vocabulary.
- Uses dedicated checkpoints directory (`models/krio-llm/checkpoints-cpt/`).

### Run Tracking
Each stage writes a machine-readable run record committed to `runs/<run_id>.json` (metadata only, no training data):
- Git commit hash
- Lock hash
- Dataset manifest ID (`manifest_id` + SHA-256)
- Model repository & revision
- Random seed
- Hardware info & metrics (chrF++, BLEU, eval loss)

---

## Base Model Selection Criteria

Evaluate each candidate model on:

| Criterion | Why it matters |
|---|---|
| Krio tokenisation efficiency | Poor tokenisation inflates sequence length and training cost |
| Multilingual pretraining | Better starting point for low-resource languages |
| Licence | Must allow research and eventual commercial use |
| Hardware requirements | Must fit available GPU memory |
| Community support | Active maintenance reduces engineering burden |

See `src/sierraleone_ai/models/registry.py` for current candidate list (separated by `task`: `chat_llm`, `translation_seq2seq`, `asr`, `tts`).

---

## Tokenisation Analysis (Required Before Training — SLAI-029)

Use the built-in reproducible CLI command (requires `pip install -e ".[training]"`; **downloads tokenizer files only, never model weights**):

```bash
slai audit-tokenizer \
  --tokenizer Qwen/Qwen3-4B-Base \
  --corpus datasets/krio/validated/krio_text_train.jsonl \
  --min-sentences 1000
```

This reports:
- `tokens_per_word` and `tokens_per_character`
- `unknown_token_rate`
- `top_fragmented_words` (fragmentation of the 20 most frequent words)
- `open_vowel_handling` (how `ɛ` and `ɔ` are tokenised)
- `english_sequence_length_overhead` (token ratio versus English on parallel sentences)

> **Note on fertility targets:** Any `≤ 2.0 tokens/word` rule of thumb is an **unvalidated heuristic**, not a universal pass/fail threshold. Treat tokenizer fertility as one selection criterion alongside zero-shot Krio benchmark performance, `ɛ`/`ɔ` handling, VRAM budget, and licence terms.

---

## Training Readiness & Dry-Run Inspection (Implemented)

```bash
# Inspect dataset readiness and print the dry-run plan (downloads nothing, starts nothing)
slai check-readiness krio --dry-run
```

---

## Training Pipeline (**planned – not implemented**)

### Phase A: Data Preparation (**planned – not implemented**)
```bash
# (planned – not implemented)
slai prepare-data --language krio --split train
```

### Phase B: Continued Pre-training (**planned – not implemented**)
```bash
# (planned – not implemented – requires SLAI_ALLOW_TRAINING=true + approval manifest)
slai train continued-pretraining \
  --model MODEL_ID \
  --data datasets/krio/validated/ \
  --output models/krio-llm/checkpoints-cpt/
```

### Phase C: Instruction Fine-tuning (**planned – not implemented**)
```bash
# (planned – not implemented)
slai train fine-tune \
  --model models/krio-llm/checkpoints-cpt/ \
  --data datasets/krio/conversations/ \
  --method qlora_4bit \
  --output models/krio-llm/checkpoints-sft/
```

### Phase D: Evaluation (**planned – not implemented**)
```bash
# (planned – not implemented)
slai evaluate \
  --model models/krio-llm/checkpoints-sft/ \
  --benchmark benchmarks/krio/
```

---

## Hardware & OS Requirements (Estimates)

| Model size | Minimum VRAM (inference) | Minimum VRAM (LoRA fine-tune) |
|---|---|---|
| ≤1B parameters | 4 GB | 8 GB |
| 1–3B parameters | 8 GB | 16 GB |
| 3–8B parameters | 16 GB | 24 GB |
| 8–13B parameters | 24 GB | 40 GB |

> **Operating System Note (SLAI-018):** `bitsandbytes` 4-bit/8-bit quantisation is unreliable on native Windows. Always run QLoRA/LoRA training on **Linux**, **WSL2 (Ubuntu)**, or a **Linux cloud GPU instance**. Also note that NVIDIA T4 GPUs (e.g. default Colab) do not natively accelerate `bfloat16`; our default config uses `float16` for broad compatibility.

---

## Important Warnings

- **Never train on `raw/` folders directly.**
- **Never train on `machine_generated` data without clear labelling.**
- **Never mix Nigerian Pidgin, Liberian English or other creoles into Krio training data.**
- **Always checkpoint regularly** to avoid losing training progress.
- **Always evaluate on the held-out benchmark** before claiming model quality.
- **Document every training run** with a signed `training/approvals/<run_id>.json` manifest.
