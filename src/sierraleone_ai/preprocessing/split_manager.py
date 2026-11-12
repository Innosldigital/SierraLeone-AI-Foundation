"""
Dataset split management (SLAI-009 / SLAI-023).

Ensures that:
- Train / validation / test splits are created consistently using largest-remainder
  with guaranteed non-empty active splits when len(items) >= number of active splits.
- Speaker leakage is prevented in audio datasets: records without `speaker_id` fail
  closed with `ValueError` (no random fallback), each speaker is assigned to at most
  one split, held-out speakers are permanently excluded from train/tuning splits,
  and speaker-to-split manifests can be persisted and reused when new data arrives.
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, TypeVar

from loguru import logger

from sierraleone_ai.config import settings

T = TypeVar("T")

DEFAULT_RATIOS: dict[str, float] = {"train": 0.8, "validation": 0.1, "test": 0.1}


def _validate_ratios(ratios: dict[str, float]) -> None:
    """Validate split ratios (non-empty keys, non-negative values, sum ≈ 1.0)."""
    if not ratios:
        raise ValueError("Split ratios dictionary must not be empty")
    for name, val in ratios.items():
        if not name or not name.strip():
            raise ValueError("Split names must be non-empty strings")
        if val < 0:
            raise ValueError(f"Split ratio for '{name}' must be non-negative, got {val}")
    total = sum(ratios.values())
    if abs(total - 1.0) >= 1e-9:
        raise ValueError(f"Ratios must sum to 1.0, got {total:.10f}")


def deterministic_split(
    items: list[T],
    ratios: dict[str, float] | None = None,
    seed: int = 42,
    *,
    shuffle: bool = True,
) -> dict[str, list[T]]:
    """
    Split a list of items into named splits using a fixed seed.

    Uses the largest-remainder method and guarantees at least 1 item for every
    split with ratio > 0 whenever len(items) >= number of active splits.
    Raises ValueError if 0 < len(items) < number of active splits.
    """
    ratios = ratios or DEFAULT_RATIOS
    _validate_ratios(ratios)

    names = list(ratios.keys())
    active_indices = [i for i, name in enumerate(names) if ratios[name] > 0]
    num_active = len(active_indices)
    n = len(items)

    if 0 < n < num_active:
        raise ValueError(f"need at least {num_active} speakers/items for these ratios (got {n})")

    if n == 0:
        return {name: [] for name in names}

    shuffled = list(items)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(shuffled)

    # Largest-remainder allocation
    exact = [ratios[name] * n for name in names]
    floor_sizes = [int(e) for e in exact]
    remainders = [(exact[i] - floor_sizes[i], i) for i in range(len(names))]
    deficit = n - sum(floor_sizes)
    for _, idx in sorted(remainders, reverse=True)[:deficit]:
        floor_sizes[idx] += 1

    # Guarantee >= 1 item for every split with ratio > 0 when n >= num_active (SLAI-009)
    for idx in active_indices:
        if floor_sizes[idx] == 0:
            donor_idx = max(
                range(len(names)),
                key=lambda j: (floor_sizes[j], exact[j]),
            )
            if floor_sizes[donor_idx] > 1:
                floor_sizes[donor_idx] -= 1
                floor_sizes[idx] += 1

    splits: dict[str, list[T]] = {}
    start = 0
    for name, size in zip(names, floor_sizes, strict=True):
        splits[name] = shuffled[start : start + size]
        start += size

    for name, items_in_split in splits.items():
        logger.info(f"Split '{name}': {len(items_in_split)} items")

    return splits


def default_speaker_manifest_path(language: str = "krio", version: int = 1) -> Path:
    """Return canonical path `datasets/<lang>/splits/speaker_splits_v<N>.json`."""
    return settings.language_data_dir(language) / "splits" / f"speaker_splits_v{version}.json"


def speaker_aware_split(
    records: list[dict[str, Any]],
    speaker_id_field: str = "speaker_id",
    ratios: dict[str, float] | None = None,
    seed: int = 42,
    *,
    manifest_path: Path | None = None,
    persist_manifest: bool = False,
    language: str = "krio",
    version: int = 1,
    held_out_speakers: set[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """
    Split audio records ensuring each speaker appears in only one split (SLAI-023).

    Fails closed (raises ValueError listing offending record IDs) if any record
    is missing a non-empty `speaker_id`. Reuses existing speaker→split assignments
    from `manifest_path` when present so existing speakers never move between splits,
    and permanently excludes held-out speakers from train/validation/test tuning.
    """
    ratios = ratios or DEFAULT_RATIOS
    _validate_ratios(ratios)

    if not records:
        return {name: [] for name in ratios}

    # 1. Fail closed on missing speaker_id (SLAI-023)
    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_ids: list[str] = []
    for idx, rec in enumerate(records):
        sid = rec.get(speaker_id_field)
        if not isinstance(sid, str) or not sid.strip():
            rec_id = (
                rec.get("id")
                or rec.get("record_id")
                or (rec.get("metadata", {}) or {}).get("record_id")
                or f"index_{idx}"
            )
            missing_ids.append(str(rec_id))
        else:
            by_speaker[sid.strip()].append(rec)

    if missing_ids:
        raise ValueError(
            f"speaker_aware_split failed closed: {len(missing_ids)} record(s) missing "
            f"'{speaker_id_field}'. Offending record IDs: {missing_ids}"
        )

    # 2. Load existing speaker manifest if provided and exists
    existing_map: dict[str, str] = {}
    persisted_held_out: set[str] = set(held_out_speakers or set())
    target_manifest = manifest_path or (
        default_speaker_manifest_path(language, version) if persist_manifest else None
    )
    if target_manifest is not None and target_manifest.exists():
        data = json.loads(target_manifest.read_text(encoding="utf-8"))
        existing_map = dict(data.get("speaker_to_split", {}))
        persisted_held_out.update(data.get("held_out_speakers", []))

    for spk, sp_name in existing_map.items():
        if sp_name == "held_out":
            persisted_held_out.add(spk)

    # 3. Separate existing/held-out speakers from new speakers
    all_speakers = sorted(by_speaker.keys())
    new_speakers = [
        s for s in all_speakers if s not in existing_map and s not in persisted_held_out
    ]

    speaker_to_split: dict[str, str] = dict(existing_map)
    for spk in persisted_held_out:
        speaker_to_split[spk] = "held_out"

    if new_speakers:
        active_existing = {spk: sp for spk, sp in existing_map.items() if sp in ratios}
        num_active_splits = sum(1 for r in ratios.values() if r > 0)
        if not active_existing or len(new_speakers) >= num_active_splits:
            # Shuffle once inside deterministic_split (no redundant double-shuffle: SLAI-009)
            new_splits = deterministic_split(new_speakers, ratios=ratios, seed=seed, shuffle=True)
            for split_name, split_speakers in new_splits.items():
                for spk in split_speakers:
                    speaker_to_split[spk] = split_name
        else:
            # Incremental arrival of < num_active_splits new speakers on top of an existing manifest:
            # assign each new speaker to the active split furthest below its target ratio.
            rng = random.Random(seed)
            shuffled_new = list(new_speakers)
            rng.shuffle(shuffled_new)
            counts = {
                name: sum(1 for sp in speaker_to_split.values() if sp == name) for name in ratios
            }
            for spk in shuffled_new:
                total_after = sum(counts.values()) + 1
                best_split = max(
                    (name for name, r in ratios.items() if r > 0),
                    key=lambda name: ratios[name] * total_after - counts[name],
                )
                speaker_to_split[spk] = best_split
                counts[best_split] += 1

    # 4. Assemble split records (excluding held_out from train/val/test unless 'held_out' in ratios)
    result: dict[str, list[dict[str, Any]]] = {name: [] for name in ratios}
    for spk in all_speakers:
        assigned_split = speaker_to_split[spk]
        if assigned_split == "held_out" and "held_out" not in result:
            # Permanently excluded from train/tuning splits (SLAI-023)
            continue
        if assigned_split in result:
            result[assigned_split].extend(by_speaker[spk])

    # 5. Automatic post-split speaker leakage audit (SLAI-023)
    seen_speakers_per_split: dict[str, set[str]] = {
        split_name: {r[speaker_id_field].strip() for r in split_recs}
        for split_name, split_recs in result.items()
    }
    split_names = list(seen_speakers_per_split.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            overlap = (
                seen_speakers_per_split[split_names[i]] & seen_speakers_per_split[split_names[j]]
            )
            if overlap:
                raise RuntimeError(
                    f"Speaker leakage detected between '{split_names[i]}' and "
                    f"'{split_names[j]}': {sorted(overlap)}"
                )

    # 6. Persist manifest atomically if requested (PA-022)
    if target_manifest is not None:
        target_manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest_payload = {
            "version": version,
            "language": language,
            "seed": seed,
            "ratios": ratios,
            "speaker_to_split": speaker_to_split,
            "held_out_speakers": sorted(persisted_held_out),
        }
        tmp = target_manifest.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(manifest_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(tmp, target_manifest)

    # 7. Log final counts and ratios (PA-022)
    total_recs = len(records)
    for name, split_recs in result.items():
        ratio_pct = (len(split_recs) / total_recs * 100) if total_recs else 0.0
        logger.info(
            f"Speaker-aware split '{name}': {len(split_recs)} records ({ratio_pct:.1f}%), "
            f"{len(seen_speakers_per_split.get(name, set()))} speakers"
        )

    return result
