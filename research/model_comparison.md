# Base Model Comparison for Krio Language Adaptation

**Status:** Research phase — no model has been downloaded or trained.
**Last updated:** 2026-11-10

> [!IMPORTANT]
> All licence, language-coverage, and parameter entries below must be verified
> from the official Hugging Face model card before trusting. Fields marked
> "VERIFY" are placeholders.

---

## Candidate Models

| Model | Parameters | Licence | Multilingual | Min VRAM | African Language Support | Notes |
|---|---|---|---|---|---|---|
| Aya-23 8B | 8B | CC-BY-NC-4.0 | ✅ Yes | 16 GB | **None for sub-Saharan Africa** — 23 languages, none sub-Saharan; Krio not covered | NC licence; "Aya" branding is misleading for African coverage |
| Tiny Aya / Aya-Expanse variants | ~3.35B–8B | VERIFY | ✅ Yes | 8 GB | 70+ languages; Krio (kri) coverage **unverified** | Released Feb 2026 (arXiv 2603.11510); verify model card |
| AfroLLaMA / AfroLlama_V1 | VERIFY | VERIFY | ✅ Yes | 16 GB | ~6 African languages; Krio **not listed** | Verify base model (Llama-2 or Llama-3), language list, and licence from model card |
| Llama-3.1-8B | 8B | VERIFY | ❌ Primarily EN | 16 GB | Base has none; used in AfriqueLLM 2026 for 20 African languages | Strong baseline; Krio not in AfriqueLLM languages |
| Gemma-3 4B | 4B | VERIFY (Gemma ToU) | ❌ Primarily EN | 8 GB | Base has none; used in AfriqueLLM 2026 | Lightweight; verify Terms of Use |
| Gemma-3 12B | 12B | VERIFY (Gemma ToU) | ❌ Primarily EN | 24 GB | Base has none; used in AfriqueLLM 2026 | Requires 24 GB VRAM |
| Qwen3-4B-Base | 4B | VERIFY | ✅ Yes | 8 GB | Used in AfriqueLLM 2026; Krio coverage unknown | Verify multilingual coverage |
| Qwen3-8B-Base | 8B | VERIFY | ✅ Yes | 16 GB | Used in AfriqueLLM 2026; Krio coverage unknown | Verify multilingual coverage |
| mT5-base | 580M | Apache-2.0 | ✅ 101 langs | 4 GB | 101 languages (mC4); Krio coverage unknown | Encoder-decoder; Apache-2.0 is most permissive licence here |

---

## Key Measurement Required Before Decision

For each candidate model, run tokenisation efficiency on a representative neutral sample
(do NOT use Krio sentences — use English test text until native speakers provide verified Krio):

```python
# Run after: pip install transformers (requires network)
from transformers import AutoTokenizer

# Use NEUTRAL English test text until native-speaker Krio is available
TEST_TEXT = "The farmer is preparing the field for the planting season."

for repo in [
    "CohereForAI/aya-23-8B",
    "meta-llama/Llama-3.1-8B",
    "google/gemma-3-4b-pt",
    "Qwen/Qwen3-4B-Base",
    "google/mt5-base",
]:
    tokenizer = AutoTokenizer.from_pretrained(repo)
    tokens = tokenizer.encode(TEST_TEXT)
    print(f"{repo}: {len(tokens)} tokens / {len(TEST_TEXT.split())} words "
          f"= {len(tokens)/len(TEST_TEXT.split()):.2f} tokens/word")
```

Target: ≤ 2 tokens per average word. This measurement MUST be repeated with
verified native-speaker Krio text once it is available.

---

## Context from Recent Research

| Study | Key finding | Implication |
|---|---|---|
| AfriqueLLM (arXiv 2601.06395, ACL 2026) | CPT on 26B tokens across 20 African languages using Llama-3.1-8B, Gemma-3 (4B/12B), Qwen3 (4B/8B/14B) | A 50,000-sentence Krio corpus (~1M tokens) is ~4 orders of magnitude smaller than what benefited CPT; consider SFT/RAG-first strategy |
| AfriqueLLM | Machine-translated data from high-resource languages helped CPT | Policy option: labelled `machine_generated` data for CPT-only use — human decision required |
| Lugha-Llama (arXiv 2504.06536) | Used ~10B tokens; describes AfroLlama as Llama-3-8B-based | AfroLLaMA registry entry needs verification |
| "Benchmarking ASR for African Languages" (arXiv 2512.10968) | Most ASR gains in first ~50 h of fine-tuning data | 20–50 h of clean, consented, speaker-diverse audio is a realistic first ASR milestone |
| Google Translate | Krio added May 2022 (zero-shot, monolingual) | A free baseline already exists; your model must beat it on YOUR benchmark |
| LINGUA Africa 2026 | *Krio Voices* (Christex Foundation, Sierra Leone) and *SAM App* (Institute for Development, Sierra Leone) awarded funding | Potential data/benchmark collaboration — contact before collecting |

---

## Recommendation

**No model is recommended yet.** The selection decision requires:

1. ☐ Tokenisation fertility audit completed on verified Krio text (not invented text)
2. ☐ Licence compatibility confirmed from official model card
3. ☐ Zero-shot baseline measured on the held-out benchmark
4. ☐ Hardware and training cost confirmed
5. ☐ Owner approval granted

Do not set `recommended=True` in `models/registry.py` until all 5 items are checked.
