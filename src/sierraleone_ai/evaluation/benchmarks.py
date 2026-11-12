"""
Held-out benchmark protection utilities.

The benchmarks directory contains permanently held-out evaluation sets that
MUST NEVER be used for training. This module provides:

1.  `benchmark_fingerprints(language)` — SHA-256 fingerprints of every
    real benchmark record's text (case/punctuation/whitespace normalised).

2.  `find_benchmark_overlap(texts, language)` — returns indices of texts
    whose fingerprint matches any benchmark record.

3.  `assert_held_out_integrity(language)` — returns False (and logs a warning)
    if the benchmark directory has no real (non-template) records yet.

NOTE: exact SHA-256 matching detects identical text modulo case/punctuation.
      It does NOT detect near-duplicates or paraphrases.
      TODO: add MinHash / n-gram Jaccard when real data exists.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from loguru import logger

from sierraleone_ai.config import settings
from sierraleone_ai.datasets.constants import TEMPLATE_PREFIXES as _TEMPLATE_PREFIXES

_SCHEMA_TEMPLATE_STEMS = ("held_out_eval_schema", "example_")


def _normalise_for_fingerprint(text: str) -> str:
    """
    Normalise text before fingerprinting so minor variations are caught:
    NFC, casefolded, punctuation stripped, whitespace collapsed.
    """
    t = unicodedata.normalize("NFC", text)
    t = t.casefold()
    t = re.sub(r"[^\w\s]", "", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def text_fingerprint(text: str) -> str:
    """Return a SHA-256 hex digest of the normalised text."""
    return hashlib.sha256(_normalise_for_fingerprint(text).encode("utf-8")).hexdigest()


def iter_texts(obj: Any) -> Iterator[str]:
    """
    Iterate over all text-bearing fields in a record (dict or model).
    Used for benchmark extraction, contamination detection, and data loading checks.
    """
    if hasattr(obj, "model_dump"):
        d = obj.model_dump()
    elif isinstance(obj, dict):
        d = obj
    else:
        d = getattr(obj, "__dict__", {})

    for k in (
        "text",
        "source_text",
        "target_text",
        "question",
        "answer",
        "transcript",
        "prompt",
        "reference",
        "references",
    ):
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            yield v
        elif isinstance(v, list):
            for x in v:
                if isinstance(x, str) and x.strip():
                    yield x
    for t in d.get("turns", []) or []:
        if isinstance(t, dict) and isinstance(t.get("text"), str) and t["text"].strip():
            yield t["text"]
        elif hasattr(t, "text") and isinstance(t.text, str) and t.text.strip():
            yield t.text


def _is_template_record(source: str) -> bool:
    return source.upper().startswith(_TEMPLATE_PREFIXES)


def _extract_texts_from_jsonl(path: Path) -> list[str]:
    """Extract all text fields from a JSONL benchmark file."""
    texts: list[str] = []
    with path.open(encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rec = json.loads(line)
                meta = rec.get("metadata", {})
                if _is_template_record(meta.get("source", "")):
                    continue
                for t in iter_texts(rec):
                    texts.append(t)
            except (json.JSONDecodeError, KeyError):
                continue
    return texts


def benchmark_fingerprints(language: str = "krio") -> set[str]:
    """
    Return SHA-256 fingerprints for all real benchmark texts for a given language.
    Template records are excluded.
    """
    bench_dir = settings.language_benchmark_dir(language)
    if not bench_dir.exists():
        return set()

    fps: set[str] = set()
    for jf in bench_dir.glob("*.jsonl"):
        if any(jf.stem.startswith(s) for s in _SCHEMA_TEMPLATE_STEMS):
            continue
        for text in _extract_texts_from_jsonl(jf):
            fps.add(text_fingerprint(text))

    return fps


def find_benchmark_overlap(
    texts: Iterable[str],
    language: str = "krio",
) -> list[int]:
    """
    Return the indices of texts that match a benchmark record fingerprint.

    Use this in the data-preparation pipeline to prevent training data
    from leaking into the held-out evaluation set.
    """
    fps = benchmark_fingerprints(language)
    if not fps:
        logger.debug("No benchmark fingerprints loaded — overlap check skipped.")
        return []
    return [i for i, t in enumerate(texts) if text_fingerprint(t) in fps]


def assert_held_out_integrity(language: str = "krio") -> bool:
    """
    Verify that the held-out benchmark directory exists and contains
    at least one real (non-template) record.

    Returns True if the benchmark is in place, False otherwise.
    """
    bench_dir = settings.language_benchmark_dir(language)

    if not bench_dir.exists():
        logger.warning(
            f"[BENCHMARK] Directory not found: {bench_dir}. "
            f"Create held-out evaluation data before training. "
            f"See documentation/DATA_COLLECTION_GUIDE.md."
        )
        return False

    fps = benchmark_fingerprints(language)
    if not fps:
        logger.warning(
            f"[BENCHMARK] {bench_dir} exists but contains no real (non-template) records. "
            f"The held-out benchmark must be populated by native speakers before training."
        )
        return False

    logger.info(f"[BENCHMARK] {language}: {len(fps)} text fingerprints loaded.")
    return True
