"""
Training safety guard (SLAI-002 / SLAI-024).

Prevents accidental expensive GPU training jobs from starting.
Training requires ALL of:
  1. SLAI_ALLOW_TRAINING=true in .env / environment
  2. A verified machine-readable approval manifest (`training/approvals/<run_id>.json`)
     with base model revision, licence review, dataset versions + SHA-256 hashes
     (never in `raw/`), split manifest ID, benchmark ID, GPU/cost estimate, and approver
  3. Minimum ELIGIBLE-record thresholds met, read from a hash-verified dataset
     manifest (`datasets/manifests/<language>-<split>-vN.manifest.json`, PA-001)
  4. Held-out evaluation benchmark integrity verified
  5. A base model selected in the registry (`models/selected_base_model.json`)

Supports `dry_run=True` which inspects and returns the full data/model/hardware plan
while downloading nothing and starting nothing.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field, field_validator

from sierraleone_ai.config import settings

# ---------------------------------------------------------------------------
# Minimum dataset thresholds (versioned in training/readiness_thresholds.json)
# ---------------------------------------------------------------------------
# Fail-closed defaults, used only if the JSON file is missing. They must match it.
_DEFAULT_MIN_RECORDS: dict[str, int] = {
    "cleaned_text": 50_000,
    "translations": 10_000,
    "conversations": 1_000,
    "held_out_benchmark": 1_000,
    "asr_audio": 1_000,            # verified ASR speech-transcript pairs
    "tts_audio": 500,              # TTS recordings with voice_synthesis consent
}

# Readiness key -> training task counted in the dataset manifest (PA-001).
_READINESS_TASKS: dict[str, str] = {
    "cleaned_text": "text_training",
    "translations": "translation_training",
    "conversations": "conversation_training",
    "asr_audio": "asr_training",
    "tts_audio": "tts_training",
}


def _load_thresholds() -> dict[str, int]:
    path = settings.project_root / "training" / "readiness_thresholds.json"
    thresholds = dict(_DEFAULT_MIN_RECORDS)
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            for key, value in data.get("thresholds", {}).items():
                if key in thresholds and isinstance(value, int) and value > 0:
                    thresholds[key] = value
        except (OSError, ValueError) as exc:
            logger.warning(f"[TRAINING GUARD] Could not read {path}: {exc}; using defaults")
    return thresholds


_MIN_RECORDS: dict[str, int] = _load_thresholds()

_SHA256_RE = re.compile(r"^[a-fA-F0-9]{64}$")


@dataclass
class ReadinessItem:
    key: str
    have: int
    need: int
    detail: str = ""

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
        suffix = f" ({self.detail})" if self.detail else ""
        return f"  {status} {self.key}: {self.have:,} / {self.need:,} records{suffix}"


class ModelLicenceReview(BaseModel):
    """Record of human verification of base model licence terms."""

    reviewer: str = Field(..., min_length=1)
    reviewed_on: str = Field(..., min_length=1)
    licence: str = Field(..., min_length=1)
    commercial_permitted: bool = False


class TrainingApprovalManifest(BaseModel):
    """
    Machine-readable training approval manifest (`training/approvals/<run_id>.json`) (SLAI-024).
    """

    run_id: str = Field(..., min_length=1)
    task: str = Field(..., min_length=1)
    base_model_id: str = Field(..., min_length=1)
    base_model_revision: str = Field(..., min_length=1, description="Exact HF commit SHA or tag")
    model_licence_review: ModelLicenceReview
    dataset_versions: dict[str, str] = Field(..., min_length=1)
    dataset_sha256_hashes: dict[str, str] = Field(
        ...,
        min_length=1,
        description="Map of dataset file path -> 64-char SHA-256 hex digest",
    )
    split_manifest_id: str = Field(..., min_length=1)
    benchmark_id: str = Field(..., min_length=1)
    estimated_gpu_hours: float = Field(..., gt=0)
    estimated_cost_usd: float = Field(..., ge=0)
    approver: str = Field(..., min_length=1)

    @field_validator("base_model_id")
    @classmethod
    def reject_tbd_model(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned or cleaned.upper().startswith("TBD"):
            raise ValueError(f"base_model_id must be a resolved model identifier, got '{v}'")
        return cleaned

    @field_validator("dataset_sha256_hashes")
    @classmethod
    def validate_dataset_paths_and_hashes(cls, v: dict[str, str]) -> dict[str, str]:
        if not v:
            raise ValueError("dataset_sha256_hashes must not be empty")
        for path_str, sha in v.items():
            parts = PurePosixPath(path_str.replace("\\", "/")).parts
            if "raw" in parts:
                raise ValueError(
                    f"Training manifest references a 'raw/' path ('{path_str}'). "
                    "No training code may read from raw/ folders directly."
                )
            if not _SHA256_RE.match(sha):
                raise ValueError(
                    f"Invalid SHA-256 digest for '{path_str}': expected 64 hex chars, got '{sha}'"
                )
        return v


def verify_training_manifest(
    manifest_path: Path,
    base_dir: Path | None = None,
) -> TrainingApprovalManifest:
    """
    Load and verify a TrainingApprovalManifest on disk, checking that every
    referenced dataset file exists (outside `raw/`) and matches its SHA-256 hash.
    """
    if not manifest_path.exists():
        raise RuntimeError(
            f"[TRAINING GUARD] Approval manifest file not found: {manifest_path}"
        )
    try:
        raw_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = TrainingApprovalManifest(**raw_data)
    except Exception as exc:
        raise RuntimeError(
            f"[TRAINING GUARD] Invalid training approval manifest '{manifest_path}': {exc}"
        ) from exc

    root = base_dir or settings.project_root
    for rel_or_abs, expected_sha in manifest.dataset_sha256_hashes.items():
        candidate = Path(rel_or_abs)
        target = candidate if candidate.is_absolute() else (root / candidate)
        if not target.exists() or not target.is_file():
            raise RuntimeError(
                f"[TRAINING GUARD] Dataset file referenced in manifest is missing: '{rel_or_abs}'"
            )
        actual_sha = hashlib.sha256(target.read_bytes()).hexdigest().lower()
        if actual_sha != expected_sha.lower():
            raise RuntimeError(
                f"[TRAINING GUARD] Dataset SHA-256 mismatch for '{rel_or_abs}': "
                f"expected {expected_sha.lower()}, got {actual_sha}"
            )
    return manifest


def check_dataset_readiness(
    language: str = "krio",
    split: str = "train",
    manifest_path: Path | None = None,
) -> dict[str, ReadinessItem]:
    """
    Check whether minimum ELIGIBLE-record thresholds are met before training (PA-001).

    Counts come from the dataset manifest built by `slai build-manifest`, which uses the
    same strict loader + EligibilityPolicy as training, de-duplicated by record_id.
    The manifest's file hashes are re-verified here; if there is no manifest, or the data
    on disk has changed since it was built, every data count is reported as 0.
    """
    from sierraleone_ai.datasets.manifest import (
        latest_manifest_path,
        load_manifest,
        verify_manifest,
    )
    from sierraleone_ai.evaluation.benchmarks import count_validated_benchmark_items

    counts: dict[str, int] = {}
    detail = ""
    path = manifest_path or latest_manifest_path(language, split)
    if path is None:
        detail = f"no dataset manifest; run `slai build-manifest {language} --split {split}`"
    else:
        try:
            manifest = load_manifest(path)
        except (OSError, ValueError) as exc:
            detail = f"invalid manifest {path.name}: {exc}"
        else:
            if manifest.language != language or manifest.split != split:
                detail = f"manifest {manifest.manifest_id} is not for {language}/{split}"
            else:
                problems = verify_manifest(manifest)
                if problems:
                    detail = f"manifest {manifest.manifest_id} is stale: {problems[0]}"
                    if len(problems) > 1:
                        detail += f" (+{len(problems) - 1} more)"
                else:
                    counts = manifest.eligible_counts

    items = {
        key: ReadinessItem(key, int(counts.get(task, 0)), _MIN_RECORDS[key], detail)
        for key, task in _READINESS_TASKS.items()
    }
    items["held_out_benchmark"] = ReadinessItem(
        "held_out_benchmark",
        count_validated_benchmark_items(language),
        _MIN_RECORDS["held_out_benchmark"],
    )
    order = ("cleaned_text", "translations", "conversations", "held_out_benchmark", "asr_audio", "tts_audio")
    return {k: items[k] for k in order}


def require_training_approval(
    task: str = "model training",
    language: str = "krio",
    manifest_path: Path | None = None,
    *,
    check_readiness: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Raise RuntimeError if training prerequisites are not met (unless dry_run=True).

    When `dry_run=True`, inspects and returns the structured training readiness plan
    without downloading weights, writing files, or starting training (SLAI-024).
    """
    from sierraleone_ai.evaluation.benchmarks import assert_held_out_integrity
    from sierraleone_ai.models.registry import get_recommended

    readiness = check_dataset_readiness(language)
    recommended = get_recommended()
    held_out_ok = assert_held_out_integrity(language)

    plan: dict[str, Any] = {
        "dry_run": dry_run,
        "task": task,
        "language": language,
        "allow_training_env": settings.allow_training,
        "manifest_path": str(manifest_path) if manifest_path else None,
        "selected_base_model": recommended.hf_repo if recommended else None,
        "held_out_integrity_ok": held_out_ok,
        "readiness": {
            k: {"have": v.have, "need": v.need, "ok": v.ok}
            for k, v in readiness.items()
        },
        "ready_to_train": False,
    }

    if dry_run:
        logger.info(f"[TRAINING GUARD DRY-RUN] Plan for '{task}': downloads=0, writes=0")
        return plan

    # 1. Environment gate
    if not settings.allow_training:
        raise RuntimeError(
            f"[TRAINING GUARD] '{task}' is blocked.\n"
            f"Set SLAI_ALLOW_TRAINING=true in .env to proceed — but this alone is insufficient.\n"
            f"All prerequisites (approval manifest, dataset thresholds, base model) must also pass.\n"
        )

    # 2. If a manifest_path is explicitly provided, validate it immediately
    if manifest_path is not None:
        verify_training_manifest(manifest_path)

    # 3. Dataset readiness
    if check_readiness:
        failures = [str(item) for item in readiness.values() if not item.ok]
        if failures:
            raise RuntimeError(
                f"[TRAINING GUARD] '{task}' blocked — dataset not ready:\n"
                + "\n".join(failures)
                + "\n\nSee documentation/DATA_COLLECTION_GUIDE.md for collection instructions."
            )

    # 4. Require machine-readable approval manifest (SLAI-024)
    if manifest_path is None:
        raise RuntimeError(
            f"[TRAINING GUARD] '{task}' blocked — missing approval manifest.\n"
            "Provide a verified `training/approvals/<run_id>.json` manifest path "
            "containing base_model_id, base_model_revision, model_licence_review, "
            "dataset_versions, dataset_sha256_hashes, split_manifest_id, benchmark_id, "
            "estimated_gpu_hours, estimated_cost_usd, and approver."
        )

    # 5. Held-out benchmark integrity
    if check_readiness and not held_out_ok:
        raise RuntimeError(
            f"[TRAINING GUARD] '{task}' blocked — held-out benchmark integrity check failed."
        )

    # 6. Base model selection
    if recommended is None:
        raise RuntimeError(
            f"[TRAINING GUARD] '{task}' blocked — no base model selected.\n"
            f"Complete the tokenisation audit in research/model_comparison.md,\n"
            f"record the decision in models/selected_base_model.json,\n"
            f"and get approval before training."
        )

    plan["ready_to_train"] = True
    logger.warning(f"[TRAINING GUARD] All prerequisites met. Starting: {task}")
    return plan
