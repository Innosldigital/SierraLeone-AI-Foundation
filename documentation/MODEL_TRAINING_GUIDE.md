# Model Training Guide

> ⚠️ **No training should begin until:**
> 1. The dataset is reviewed and approved.
> 2. A base model has been selected.
> 3. Hardware resources are confirmed.
> 4. `SLAI_ALLOW_TRAINING=true` is set in `.env`.

---

## Strategy

We are NOT training a large language model from randomly initialised weights.

**Approach:**
1. Select a suitable open-weight multilingual base model (see Model Research below).
2. Apply **continued pre-training** on Krio text to adapt the model's vocabulary and knowledge.
3. Apply **instruction fine-tuning** on conversations, Q&A pairs and translation pairs.
4. Use **LoRA/QLoRA** for parameter-efficient fine-tuning to reduce hardware requirements.
5. Evaluate with native speakers before any claim of fluency.

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

See `src/sierraleone_ai/models/registry.py` for current candidate list.

---

## Tokenisation Analysis (Required Before Training)

For each candidate model, measure:

```python
# Example – run this after installing transformers
from transformers import AutoTokenizer

KRIO_SAMPLE = "Mi de go maket tumara mɔning fɔ bai som rais en fish."

tokenizer = AutoTokenizer.from_pretrained("MODEL_HF_REPO")
tokens = tokenizer.encode(KRIO_SAMPLE)
print(f"Token count: {len(tokens)}")
print(f"Tokens/character: {len(tokens)/len(KRIO_SAMPLE):.2f}")
```

A model that uses many tokens per Krio word is not ideal.
Target: ≤ 2 tokens per average Krio word.

---

## Training Pipeline (Planned)

### Phase A: Data Preparation
```bash
# (future command)
slai prepare-data --language krio --split train
```

### Phase B: Continued Pre-training
```bash
# (future command – requires SLAI_ALLOW_TRAINING=true)
slai train continued-pretraining \
  --model MODEL_ID \
  --data datasets/krio/cleaned/ \
  --output models/krio-llm/checkpoint-pretrain/
```

### Phase C: Instruction Fine-tuning
```bash
# (future command)
slai train fine-tune \
  --model models/krio-llm/checkpoint-pretrain/ \
  --data datasets/krio/conversations/ \
  --method lora \
  --output models/krio-llm/checkpoint-sft/
```

### Phase D: Evaluation
```bash
# (future command)
slai evaluate \
  --model models/krio-llm/checkpoint-sft/ \
  --benchmark benchmarks/krio/
```

---

## Hardware Requirements (Estimates)

| Model size | Minimum VRAM (inference) | Minimum VRAM (LoRA fine-tune) |
|---|---|---|
| ≤1B parameters | 4 GB | 8 GB |
| 1–3B parameters | 8 GB | 16 GB |
| 3–8B parameters | 16 GB | 24 GB |
| 8–13B parameters | 24 GB | 40 GB |

Cloud GPU options to consider (not endorsed):
- Google Colab Pro (T4/A100)
- Vast.ai (cost-effective spot instances)
- RunPod
- Google Cloud / AWS (for production training)

---

## Important Warnings

- **Never train on `machine_generated` data without clear labelling.**
- **Never mix Nigerian Pidgin, Liberian English or other creoles into Krio training data.**
- **Always checkpoint regularly** to avoid losing training progress.
- **Always evaluate on the held-out benchmark** before claiming model quality.
- **Document every training run** with hyperparameters, data version and hardware used.
