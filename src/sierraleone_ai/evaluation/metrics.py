"""
Automated evaluation metrics.

IMPORTANT: Automated metrics are supplementary only.
Native speaker evaluation is mandatory before claiming model quality.
See MODEL_EVALUATION_GUIDE.md for the full evaluation protocol.

For production use consider sacrebleu (BLEU, chrF++) and jiwer (WER/CER)
which handle tokenisation edge cases more robustly.
"""
from __future__ import annotations

import math
import unicodedata
from collections import Counter


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _edit_distance(a: list, b: list) -> int:
    """Levenshtein distance between two sequences."""
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[n][m]


def bleu_score(
    references: list[list[str]],
    hypothesis: list[str],
    max_n: int = 4,
) -> float:
    """
    Sentence-level BLEU score (simplified – use sacrebleu for production).

    Differences from NLTK/sacrebleu:
    - Caps effective n-gram order at ``len(hypothesis)`` to avoid
      returning 0.0 for short hypotheses with no higher-order matches.
    - Uses add-1 (Laplace) smoothing for n-gram orders with zero clipped count.

    Args:
        references: List of reference tokenised sentences.
        hypothesis: Hypothesis tokenised sentence.
        max_n: Maximum n-gram order.

    Returns:
        BLEU score in [0, 1].
    """
    if not hypothesis:
        return 0.0

    # Cap effective n so short hypotheses can still score
    n_eff = min(max_n, len(hypothesis))

    # Brevity penalty
    ref_lens = [len(ref) for ref in references]
    closest_ref_len = min(ref_lens, key=lambda rl: abs(rl - len(hypothesis)))
    bp = 1.0 if len(hypothesis) >= closest_ref_len else math.exp(
        1 - closest_ref_len / len(hypothesis)
    )

    log_sum = 0.0
    for n in range(1, n_eff + 1):
        hyp_ngrams = _ngrams(hypothesis, n)

        clipped_count = 0
        for ngram, count in hyp_ngrams.items():
            max_ref_count = max(
                _ngrams(ref, n).get(ngram, 0) for ref in references
            )
            clipped_count += min(count, max_ref_count)

        total = sum(hyp_ngrams.values())
        # Add-1 smoothing: avoids log(0) for unmatched higher-order n-grams
        log_sum += math.log((clipped_count + 1) / (total + 1))

    return bp * math.exp(log_sum / n_eff)


def character_error_rate(reference: str, hypothesis: str) -> float:
    """
    Character Error Rate (CER) for ASR evaluation.

    Returns:
        0.0 if both strings are empty.
        0.0 if reference == hypothesis.
        float('inf') if reference is empty and hypothesis is not.
        CER in (0, inf) otherwise. Lower is better.
    """
    # Normalise to NFC before comparison
    reference = unicodedata.normalize("NFC", reference)
    hypothesis = unicodedata.normalize("NFC", hypothesis)

    if not reference and not hypothesis:
        return 0.0
    if not reference:
        return float("inf")

    r, h = list(reference), list(hypothesis)
    return _edit_distance(r, h) / len(r)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """
    Word Error Rate (WER) for ASR evaluation.

    Returns:
        0.0 if both strings are empty.
        0.0 if reference == hypothesis.
        float('inf') if reference is empty and hypothesis is not.
        WER in (0, inf) otherwise. Lower is better.
    """
    # Normalise to NFC before splitting
    reference = unicodedata.normalize("NFC", reference)
    hypothesis = unicodedata.normalize("NFC", hypothesis)

    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if not ref_words and not hyp_words:
        return 0.0
    if not ref_words:
        return float("inf")

    return _edit_distance(ref_words, hyp_words) / len(ref_words)
