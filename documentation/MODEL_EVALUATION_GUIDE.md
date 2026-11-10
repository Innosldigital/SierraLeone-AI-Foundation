# Model Evaluation Guide

Automated metrics alone are **insufficient** to evaluate Krio language quality.
Native speaker evaluation is **mandatory** before any public claims about model fluency.

---

## Evaluation Dimensions

| Dimension | Automated | Native Speaker |
|---|---|---|
| Translation adequacy | BLEU (supplementary) | ✅ Required |
| Translation fluency | — | ✅ Required |
| Grammatical correctness | — | ✅ Required |
| Vocabulary naturalness | — | ✅ Required |
| Cultural appropriateness | — | ✅ Required |
| ASR word error rate | WER ✅ | Spot check required |
| ASR character error rate | CER ✅ | Spot check required |

---

## Automated Metrics

### BLEU (Translation)
- Use `sacrebleu` for standardised BLEU scores.
- Do NOT report BLEU as the sole quality metric.
- Krio BLEU scores cannot be compared to English BLEU baselines.

### WER / CER (ASR)
- Word Error Rate for overall ASR performance.
- Character Error Rate for character-level accuracy.
- Both are implemented in `src/sierraleone_ai/evaluation/metrics.py`.

---

## Native Speaker Evaluation Protocol

### Evaluator Requirements
- Native Sierra Leonean Krio speaker
- Born or raised in Sierra Leone
- Not the same person who created the training data
- Minimum 2 independent evaluators per evaluation batch

### Evaluation Form
See `src/sierraleone_ai/evaluation/native_speaker.py` for the structured form.

Evaluators rate each output on:
1. **Grammatical correctness** (1–5)
2. **Vocabulary appropriateness** (1–5)
3. **Naturalness** (1–5)
4. **Cultural appropriateness** (1–5)
5. **Task completion** (1–5)

Evaluators must flag:
- Non-Krio creole content (Nigerian Pidgin, Liberian English, etc.)
- Factually incorrect content
- Culturally inappropriate content

### Minimum Evaluation Sample
- 200 generated outputs per evaluation round
- 100 translation pairs per evaluation round
- Drawn from the held-out benchmark only

---

## Held-Out Benchmark

The held-out evaluation set (`benchmarks/krio/`) is **permanently reserved**.
It must NEVER be used for training, even partially.

Contamination check: when training data is collected, run a deduplication
check against the benchmark set before adding new data to the training set.

---

## Reporting Standards

When reporting evaluation results, always include:

1. Model identifier and version
2. Evaluation date
3. Benchmark dataset version
4. Number of native speaker evaluators
5. Automated metric scores (with caveats)
6. Human evaluation scores (mean ± standard deviation)
7. Known failure modes
8. What the model CANNOT do

**Never claim the model is fluent based only on automated metrics.**
