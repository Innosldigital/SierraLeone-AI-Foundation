"""
Training safety guard.

Prevents accidental expensive GPU training jobs from starting.
Training requires ALL of:
  1. SLAI_ALLOW_TRAINING=true in .env
  2. Minimum dataset thresholds met (real, non-template records)
  3. A base model selected in the registry

Call `require_training_approval()` at the top of every training entry point.
"""
from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from sierraleone_ai.config import settings

# ---------------------------------------------------------------------------
# Minimum dataset thresholds (from DATA_COLLECTION_GUIDE.md)
# ---------------------------------------------------------------------------
_MIN_RECORDS: dict[str, int] = {
    "cleaned_text": 50_000,
    "translations": 10_000,
    "conversations": 1_000,
    "validated_text": 1_000,       # held-out evaluation benchmark
}


@dataclass
class ReadinessItem:
    key: str
    have: int
    need: int

    @property
    def ok(self) -> bool:
        return self.have >= self.need

    # Aliases used by test suite and CLI
    @property
    def name(self) -> str:
        return self.key

    @property
    def ready(self) -> bool:
        return self.ok

    @property
    def count(self) -> int:
        return self.have

    def __str__(self) -> str:
        status = "✓" if self.ok else "✗"
        return f"  {status} {self.key}: {self.have:,} / {self.need:,} records"


def check_dataset_readiness(language: str = "krio") -> dict[str, ReadinessItem]:
    """
    Check whether minimum real-data thresholds are met before training.

    Returns a dict of ReadinessItem (each has .ready/.ok, .count/.have, .need, .name/.key).
    Template and example files are excluded (SLAI-002).
    """
    from sierraleone_ai.datasets.loader import _count_real_records

    lang_dir = settings.language_data_dir(language)

    def _count_in(subdir: str) -> int:
        d = lang_dir / subdir
        if not d.exists():
            return 0
        total = 0
        for jf in d.glob("*.jsonl"):
            if jf.stem.startswith("example_"):
                continue
            total += _count_real_records(jf)
        return total

    cleaned = _count_in("cleaned")
    translations = _count_in("translations")
    conversations = _count_in("conversations")
    bench_dir = settings.language_benchmark_dir(language)
    benchmark = 0
    if bench_dir.exists():
        for jf in bench_dir.glob("*.jsonl"):
            if jf.stem.startswith("example_") or jf.stem.startswith("held_out_eval_schema"):
                continue
            benchmark += _count_real_records(jf)

    return {
        "cleaned_text": ReadinessItem("cleaned_text", cleaned, _MIN_RECORDS["cleaned_text"]),
        "translations": ReadinessItem("translations", translations, _MIN_RECORDS["translations"]),
        "conversations": ReadinessItem("conversations", conversations, _MIN_RECORDS["conversations"]),
        "held_out_benchmark": ReadinessItem("held_out_benchmark", benchmark, _MIN_RECORDS["validated_text"]),
    }


def require_training_approval(
    task: str = "model training",
    language: str = "krio",
    *,
    check_readiness: bool = True,
) -> None:
    """
    Raise RuntimeError if training prerequisites are not met.

    Prerequisites (ALL must pass):
      1. SLAI_ALLOW_TRAINING=true in environment / .env
      2. Minimum real-data thresholds met for `language`
      3. A base model is selected (registry.get_recommended() is not None)
    """
    # 1. Environment gate
    if not settings.allow_training:
        raise RuntimeError(
            f"[TRAINING GUARD] '{task}' is blocked.\n"
            f"Set SLAI_ALLOW_TRAINING=true in .env to proceed — but this alone is insufficient.\n"
            f"All prerequisites below must also pass.\n"
        )

    # 2. Dataset readiness
    if check_readiness:
        readiness = check_dataset_readiness(language)
        failures = [str(item) for item in readiness.values() if not item.ok]
        if failures:
            raise RuntimeError(
                f"[TRAINING GUARD] '{task}' blocked — dataset not ready:\n"
                + "\n".join(failures)
                + "\n\nSee documentation/DATA_COLLECTION_GUIDE.md for collection instructions."
            )

    # 3. Base model selection
    from sierraleone_ai.models.registry import get_recommended
    if get_recommended() is None:
        raise RuntimeError(
            f"[TRAINING GUARD] '{task}' blocked — no base model selected.\n"
            f"Complete the tokenisation audit in research/model_comparison.md,\n"
            f"set recommended=True on the chosen model in models/registry.py,\n"
            f"and get approval before training."
        )

    logger.warning(f"[TRAINING GUARD] All prerequisites met. Starting: {task}")
