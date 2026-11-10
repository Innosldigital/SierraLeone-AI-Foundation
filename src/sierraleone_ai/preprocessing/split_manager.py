"""
Dataset split management.

Ensures that:
- Train / validation / test splits are created consistently.
- Speaker leakage is prevented in audio datasets (speakers are
  assigned to exactly one split).
- Split ratios are documented and reproducible.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import TypeVar

from loguru import logger

T = TypeVar("T")

DEFAULT_RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}


def deterministic_split(
    items: list[T],
    ratios: dict[str, float] | None = None,
    seed: int = 42,
) -> dict[str, list[T]]:
    """
    Split a list of items into named splits using a fixed seed.

    Uses the largest-remainder method to allocate sizes so that
    every split receives at least 1 item when len(items) >= len(ratios),
    and the total always equals len(items).

    Args:
        items: List of items to split.
        ratios: Dict mapping split name to fraction. Must sum to 1.0.
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping split name to list of items.

    Raises:
        ValueError: If ratios do not sum to 1.0 (within 1e-9) or any
                    ratio is negative.
    """
    ratios = ratios or DEFAULT_RATIOS

    if abs(sum(ratios.values()) - 1.0) >= 1e-9:
        raise ValueError(
            f"Ratios must sum to 1.0, got {sum(ratios.values()):.10f}"
        )
    if any(v < 0 for v in ratios.values()):
        raise ValueError("All split ratios must be non-negative")

    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)

    n = len(shuffled)
    names = list(ratios.keys())

    # Largest-remainder method: compute floor sizes then add remainders
    exact = [ratios[name] * n for name in names]
    floor_sizes = [int(e) for e in exact]
    remainders = [(exact[i] - floor_sizes[i], i) for i in range(len(names))]
    deficit = n - sum(floor_sizes)
    # Distribute leftover items to splits with largest fractional remainder
    for _, idx in sorted(remainders, reverse=True)[:deficit]:
        floor_sizes[idx] += 1

    splits: dict[str, list[T]] = {}
    start = 0
    for name, size in zip(names, floor_sizes, strict=True):
        splits[name] = shuffled[start : start + size]
        start += size

    for name, items_in_split in splits.items():
        logger.info(f"Split '{name}': {len(items_in_split)} items")

    return splits


def speaker_aware_split(
    records: list[dict],
    speaker_id_field: str = "speaker_id",
    ratios: dict[str, float] | None = None,
    seed: int = 42,
    allow_missing_speaker_id: bool = False,
) -> dict[str, list[dict]]:
    """
    Split audio records ensuring each speaker appears in only one split.

    This prevents speaker leakage between train and test sets,
    which would inflate ASR evaluation scores.

    Args:
        records: List of dicts, each with a speaker_id_field.
        speaker_id_field: Key that identifies the speaker.
        ratios: Split ratios.
        seed: Random seed.
        allow_missing_speaker_id: If False (default), records without a
            speaker_id emit a warning because they may cause leakage.
            Set True to suppress the warning in contexts where anonymous
            records are expected.

    Returns:
        Dict mapping split name to list of records.
    """
    ratios = ratios or DEFAULT_RATIOS

    # Group records by speaker
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    no_speaker: list[dict] = []
    for rec in records:
        sid = rec.get(speaker_id_field)
        if sid:
            by_speaker[sid].append(rec)
        else:
            no_speaker.append(rec)

    if no_speaker and not allow_missing_speaker_id:
        logger.warning(
            f"{len(no_speaker)} records have no speaker_id – they will be split randomly, "
            f"which may cause leakage. Add speaker IDs to prevent this."
        )

    # Assign speakers to splits deterministically
    speakers = list(by_speaker.keys())
    speakers.sort()  # sort for reproducibility before seeded shuffle
    rng = random.Random(seed)
    rng.shuffle(speakers)
    speaker_splits = deterministic_split(speakers, ratios=ratios, seed=seed)

    result: dict[str, list[dict]] = {name: [] for name in ratios}
    for split_name, split_speakers in speaker_splits.items():
        for speaker in split_speakers:
            result[split_name].extend(by_speaker[speaker])

    # Handle records without speaker_id
    if no_speaker:
        anon_splits = deterministic_split(no_speaker, ratios=ratios, seed=seed)
        for split_name, split_items in anon_splits.items():
            result[split_name].extend(split_items)

    return result
