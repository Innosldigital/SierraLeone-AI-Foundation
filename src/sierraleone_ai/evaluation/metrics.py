"""
Automated evaluation metrics (SLAI-008).

IMPORTANT: Automated metrics are supplementary only.
Native speaker evaluation is mandatory before claiming model quality.
See MODEL_EVALUATION_GUIDE.md for the full evaluation protocol.

The home-grown `bleu_score` below is a sentence-level, smoothed unit-test/toy metric.
For official reporting on morphologically variable low-resource languages (such as Krio),
`chrf_score` (chrF/chrF++) is the primary automatic metric, and `sacrebleu` (BLEU + chrF++)
with its reproducibility signature should be used whenever available.
"""

from __future__ import annotations

import math
import unicodedata
from collections import Counter


def _ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _edit_distance(a: list[str], b: list[str]) -> int:
    """Levenshtein distance between two sequences using two-row DP (O(min(n,m)) space) (PA-018)."""
    if len(a) < len(b):
        a, b = b, a
    n, m = len(a), len(b)
    prev = list(range(m + 1))
    curr = [0] * (m + 1)

    for i in range(1, n + 1):
        curr[0] = i
        a_elem = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if a_elem == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev

    return prev[m]


def bleu_score(
    references: list[list[str]],
    hypothesis: list[str],
    max_n: int = 4,
) -> float:
    """
    Sentence-level, smoothed BLEU score (unit-test/toy metric — use sacrebleu
    with BLEU + chrF++ for official reporting).

    Features:
    - Raises ValueError if `references` is an empty list.
    - Caps effective n-gram order at ``min(max_n, len(hypothesis))`` so short
      hypotheses (< 4 tokens) do not artificially score 0.0.
    - Applies Lin & Och add-one smoothing for n > 1 when `clipped_count == 0`.
    """
    if not references:
        raise ValueError("references must be a non-empty list of reference token sequences")
    if max_n < 1:
        raise ValueError("max_n must be >= 1")
    if not hypothesis:
        return 0.0
    if all(len(ref) == 0 for ref in references):
        return 0.0

    # Cap effective n so short hypotheses can still score 1.0 on exact match
    n_eff = min(max_n, len(hypothesis))

    # Brevity penalty
    ref_lens = [len(ref) for ref in references if len(ref) > 0]
    closest_ref_len = min(ref_lens, key=lambda rl: abs(rl - len(hypothesis)))
    bp = (
        1.0
        if len(hypothesis) >= closest_ref_len
        else math.exp(1 - closest_ref_len / len(hypothesis))
    )

    log_sum = 0.0
    for n in range(1, n_eff + 1):
        hyp_ngrams = _ngrams(hypothesis, n)
        clipped_count = 0
        for ngram, count in hyp_ngrams.items():
            max_ref_count = max(_ngrams(ref, n).get(ngram, 0) for ref in references)
            clipped_count += min(count, max_ref_count)

        total = sum(hyp_ngrams.values())
        if clipped_count == 0:
            if n == 1:
                return 0.0
            # Add-one smoothing (Lin & Och) for n > 1
            precision = (clipped_count + 1) / (total + 1)
        else:
            precision = clipped_count / total
        log_sum += math.log(precision)

    return bp * math.exp(log_sum / n_eff)


def chrf_score(
    reference: str | list[str],
    hypothesis: str,
    max_char_n: int = 6,
    beta: float = 2.0,
) -> float:
    """
    Character n-gram F-score (chrF) — primary automatic metric for Krio and other
    morphologically variable / orthographically rich languages (SLAI-008).

    Accepts either a single reference string or a non-empty list of reference strings
    (returning the maximum chrF across references).
    Normalises strings to Unicode NFC before extracting character n-grams.
    Returns a score in [0.0, 1.0].
    """
    if isinstance(reference, list):
        if not reference:
            raise ValueError("references must not be empty")
        return max(
            chrf_score(ref, hypothesis, max_char_n=max_char_n, beta=beta) for ref in reference
        )

    ref_norm = unicodedata.normalize("NFC", reference).strip()
    hyp_norm = unicodedata.normalize("NFC", hypothesis).strip()
    if not ref_norm and not hyp_norm:
        return 1.0
    if not ref_norm or not hyp_norm:
        return 0.0

    ref_chars = list(ref_norm)
    hyp_chars = list(hyp_norm)
    n_eff = min(max_char_n, len(ref_chars), len(hyp_chars))
    if n_eff < 1:
        return 0.0

    total_p = 0.0
    total_r = 0.0
    valid_orders = 0
    for n in range(1, n_eff + 1):
        ref_ng = _ngrams(ref_chars, n)
        hyp_ng = _ngrams(hyp_chars, n)
        if not ref_ng or not hyp_ng:
            continue
        valid_orders += 1
        overlap = sum(min(c, ref_ng.get(ng, 0)) for ng, c in hyp_ng.items())
        total_p += overlap / sum(hyp_ng.values())
        total_r += overlap / sum(ref_ng.values())

    if valid_orders == 0:
        return 0.0
    avg_p = total_p / valid_orders
    avg_r = total_r / valid_orders
    if avg_p + avg_r == 0.0:
        return 0.0
    beta_sq = beta * beta
    return (1 + beta_sq) * (avg_p * avg_r) / (beta_sq * avg_p + avg_r)


def character_error_rate(reference: str, hypothesis: str) -> float:
    """
    Character Error Rate (CER) for ASR evaluation.

    Returns:
        0.0 if both strings are empty.
        0.0 if reference == hypothesis.
        float('inf') if reference is empty and hypothesis is not.
        CER in (0, inf) otherwise. Lower is better.
    """
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
    reference = unicodedata.normalize("NFC", reference)
    hypothesis = unicodedata.normalize("NFC", hypothesis)

    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if not ref_words and not hyp_words:
        return 0.0
    if not ref_words:
        return float("inf")

    return _edit_distance(ref_words, hyp_words) / len(ref_words)
