"""
Reproducible evaluation reporting with SacreBLEU (PA-018).

Provides official BLEU and chrF++ computation with signature strings.
"""

from __future__ import annotations

from typing import Any


def compute_sacrebleu(
    hypotheses: list[str],
    references: list[list[str]],
) -> dict[str, Any]:
    """
    Compute official BLEU and chrF++ scores with SacreBLEU signatures.

    Returns dict with keys:
      - 'bleu_score': float
      - 'bleu_signature': str
      - 'chrf_score': float
      - 'chrf_signature': str
    """
    try:
        import sacrebleu
    except ImportError as exc:
        raise RuntimeError(
            "Official reporting requires `sacrebleu`. Install via `pip install -e '.[eval]'`."
        ) from exc

    bleu = sacrebleu.corpus_bleu(hypotheses, references)
    chrf = sacrebleu.corpus_chrf(hypotheses, references, word_order=2)

    return {
        "bleu_score": float(bleu.score),
        "bleu_signature": str(bleu.format()),
        "chrf_score": float(chrf.score),
        "chrf_signature": str(chrf.format()),
    }
