"""
Dataset manifests and eligible-record counting (PA-001 / PA-007).

Readiness must never be computed by "counting lines in folders". Instead:

1. `scan_eligible_records()` loads every training-relevant JSONL file with the SAME
   strict loader and `EligibilityPolicy` the training code uses, per task, and
   de-duplicates by `record_id` across directories (cleaned/ + validated/ never
   double-count).
2. `build_manifest()` writes the result, plus a SHA-256 of every input file, to
   `datasets/manifests/<language>-<split>-vN.manifest.json`.
3. `training.guard.check_dataset_readiness()` reads the latest manifest and
   re-verifies the hashes, so readiness cannot drift from what training reads.

Only the explicit task folders below are scanned, so `quarantine/`, `splits/`,
`manifests/`, `raw/` and `rejected/` are never counted. `example_*` templates and
`*.issues.jsonl` quarantine reports are skipped even if placed in a task folder.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from sierraleone_ai.config import settings
from sierraleone_ai.datasets.eligibility import EligibilityPolicy, Task
from sierraleone_ai.datasets.schema import (
    AudioRecord,
    ConversationRecord,
    KrioTextRecord,
    TranslationRecord,
    VerifiedSpeechTranscriptionRecord,
)

TrainingTask = Literal[
    "text_training",
    "translation_training",
    "conversation_training",
    "asr_training",
    "tts_training",
]

TRAINING_TASKS: tuple[TrainingTask, ...] = (
    "text_training",
    "translation_training",
    "conversation_training",
    "asr_training",
    "tts_training",
)

# task -> (sub-directories under datasets/<language>/, record model)
# TTS has no separate folder: TTS-eligible data = audio records whose consent
# includes `voice_synthesis` (enforced by EligibilityPolicy(task="tts_training")).
TASK_SOURCES: dict[TrainingTask, tuple[tuple[str, ...], type[BaseModel]]] = {
    "text_training": (("cleaned", "validated"), KrioTextRecord),
    "translation_training": (("translations",), TranslationRecord),
    "conversation_training": (("conversations",), ConversationRecord),
    "asr_training": (("audio",), VerifiedSpeechTranscriptionRecord),
    "tts_training": (("audio",), AudioRecord),
}

MANIFEST_SCHEMA_VERSION: Literal[2] = 2
_MANIFEST_NAME_RE = re.compile(
    r"^(?P<lang>[a-z_]+)-(?P<split>[a-z_]+)-v(?P<n>\d+)\.manifest\.json$"
)


def is_countable_data_file(path: Path) -> bool:
    """True for real dataset JSONL files; False for templates and quarantine reports (PA-007)."""
    name = path.name
    if path.suffix != ".jsonl":
        return False
    if name.startswith("example_") or name.endswith(".issues.jsonl"):
        return False
    skip_dirs = {"quarantine", "splits", "manifests", "raw", "rejected"}
    return not any(part in skip_dirs for part in path.parts)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: Path) -> str:
    """Manifest paths are POSIX paths relative to `settings.data_dir`."""
    try:
        return path.resolve().relative_to(settings.data_dir.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _task_files(language: str, task: TrainingTask) -> list[Path]:
    subdirs, _ = TASK_SOURCES[task]
    lang_dir = settings.language_data_dir(language)
    files: list[Path] = []
    for sub in subdirs:
        d = lang_dir / sub
        if d.is_dir():
            files.extend(sorted(p for p in d.glob("*.jsonl") if is_countable_data_file(p)))
    return files


def current_data_files(language: str) -> dict[str, str]:
    """Map of every countable input file for `language` -> SHA-256."""
    seen: dict[str, str] = {}
    for task in TRAINING_TASKS:
        for p in _task_files(language, task):
            key = _rel(p)
            if key not in seen:
                seen[key] = sha256_file(p)
    return seen


@dataclass
class EligibilityScan:
    language: str
    split: str
    eligible_counts: dict[str, int] = field(default_factory=dict)
    skipped: dict[str, int] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)


def scan_eligible_records(language: str = "krio", split: str = "train") -> EligibilityScan:
    """
    Count eligible records per training task using the strict loader + EligibilityPolicy.

    Raises `DataValidationError` if any input file is malformed: a manifest must never
    be built from a dataset that does not fully validate.
    """
    from sierraleone_ai.datasets.loader import load_jsonl_records

    scan = EligibilityScan(language=language, split=split)
    skipped: Counter[str] = Counter()

    for task in TRAINING_TASKS:
        _, model_cls = TASK_SOURCES[task]
        task_name: Task = task
        policy = EligibilityPolicy(task=task_name, split=split)
        ids: set[str] = set()
        anonymous = 0
        for path in _task_files(language, task):
            report = load_jsonl_records(
                path, model_cls, split=split, policy=policy, strict=True, language=language
            )
            for reason, n in report.skipped.items():
                skipped[f"{task}:{reason}"] += n
            for rec in report.records:
                meta = getattr(rec, "metadata", None)
                rid = getattr(meta, "record_id", None)
                if rid:
                    if str(rid) in ids:
                        skipped[f"{task}:duplicate_record_id"] += 1
                        continue
                    ids.add(str(rid))
                else:
                    anonymous += 1
        scan.eligible_counts[task] = len(ids) + anonymous

    scan.skipped = dict(sorted(skipped.items()))
    scan.files = current_data_files(language)
    return scan


# ---------------------------------------------------------------------------
# Manifest model, build, discovery and verification
# ---------------------------------------------------------------------------


class DatasetManifest(BaseModel):
    schema_version: Literal[2] = MANIFEST_SCHEMA_VERSION
    language: str = Field(..., min_length=1)
    split: str = Field(..., min_length=1)
    version: int = Field(..., ge=1)
    files: dict[str, str] = Field(default_factory=dict, description="relative path -> SHA-256")
    eligible_counts: dict[str, int] = Field(default_factory=dict)
    skipped: dict[str, int] = Field(default_factory=dict)
    built_at: str
    git_commit: str | None = None

    @property
    def manifest_id(self) -> str:
        return f"{self.language}-{self.split}-v{self.version}"


def manifests_dir() -> Path:
    return settings.data_dir / "manifests"


def _existing_versions(language: str, split: str, out_dir: Path) -> list[tuple[int, Path]]:
    found: list[tuple[int, Path]] = []
    if not out_dir.is_dir():
        return found
    for p in out_dir.glob("*.manifest.json"):
        m = _MANIFEST_NAME_RE.match(p.name)
        if m and m.group("lang") == language and m.group("split") == split:
            found.append((int(m.group("n")), p))
    return sorted(found)


def latest_manifest_path(
    language: str, split: str = "train", out_dir: Path | None = None
) -> Path | None:
    versions = _existing_versions(language, split, out_dir or manifests_dir())
    return versions[-1][1] if versions else None


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=settings.project_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", sha) else None


def build_manifest(
    language: str = "krio", split: str = "train", out_dir: Path | None = None
) -> Path:
    """Scan, then atomically write `<language>-<split>-vN.manifest.json`. Returns the path."""
    target_dir = out_dir or manifests_dir()
    scan = scan_eligible_records(language, split)
    versions = _existing_versions(language, split, target_dir)
    next_version = (versions[-1][0] + 1) if versions else 1

    manifest = DatasetManifest(
        language=language,
        split=split,
        version=next_version,
        files=scan.files,
        eligible_counts=scan.eligible_counts,
        skipped=scan.skipped,
        built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),  # noqa: UP017 (py310)
        git_commit=_git_commit(),
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / f"{manifest.manifest_id}.manifest.json"
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(manifest.model_dump(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(tmp, out_path)
    return out_path


def load_manifest(path: Path) -> DatasetManifest:
    data: Any = json.loads(path.read_text(encoding="utf-8-sig"))
    return DatasetManifest.model_validate(data)


def verify_manifest(manifest: DatasetManifest) -> list[str]:
    """
    Return every reason the manifest no longer describes the data on disk
    (empty list == still valid). Detects modified, missing AND newly added files.
    """
    problems: list[str] = []
    for rel, expected in manifest.files.items():
        p = Path(rel)
        target = p if p.is_absolute() else settings.data_dir / p
        if not target.is_file():
            problems.append(f"missing:{rel}")
        elif sha256_file(target).lower() != expected.lower():
            problems.append(f"hash_mismatch:{rel}")
    for rel in sorted(set(current_data_files(manifest.language)) - set(manifest.files)):
        problems.append(f"not_in_manifest:{rel}")
    return problems
