"""
Held-out benchmark protection and integrity utilities (PA-004).

The benchmarks directory contains permanently held-out evaluation sets that
MUST NEVER be used for training. This module provides:

1.  `benchmark_fingerprints(language)` — SHA-256 fingerprints of every text-bearing
    field across all real benchmark records (case/punctuation/whitespace normalised).
2.  `find_benchmark_overlap(texts, language)` — returns indices of texts
    whose fingerprint matches any benchmark record.
3.  `build_benchmark_manifest(language)` — builds a versioned, hash-verified
    manifest of benchmark records (`benchmarks/manifests/<language>-vN.manifest.json`).
4.  `check_held_out_integrity(language)` — returns structured `{ok, count, reasons}`
    requiring valid manifest, hash match, BenchmarkItem schema, review_count >= 2,
    and >= min_benchmark_items (default 300).
5.  `assert_held_out_integrity(language)` — bool wrapper over check_held_out_integrity.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import unicodedata
from collections import Counter
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, Field

from sierraleone_ai.config import settings
from sierraleone_ai.datasets.constants import TEMPLATE_PREFIXES as _TEMPLATE_PREFIXES
from sierraleone_ai.datasets.schema import (
    BenchmarkItem,
    DatasetMetadata,
    DataSplit,
    ValidationStatus,
)

_BENCHMARK_MANIFEST_NAME_RE = re.compile(r"^(?P<lang>[a-z_]+)-v(?P<n>\d+)\.manifest\.json$")


def is_schema_template_file(path: Path) -> bool:
    """True for schema template files (e.g. held_out_eval_schema.jsonl or *schema*.jsonl)."""
    name = path.name.lower()
    return "schema" in name or name.startswith("example_")


def is_countable_benchmark_file(path: Path) -> bool:
    """True for real benchmark data JSONL files; False for templates and reports (PA-004)."""
    if path.suffix != ".jsonl":
        return False
    if is_schema_template_file(path) or path.name.endswith(".issues.jsonl"):
        return False
    skip_dirs = {"quarantine", "manifests"}
    return not any(part in skip_dirs for part in path.parts)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel_benchmark_path(path: Path) -> str:
    """Relative POSIX path under settings.benchmarks_dir."""
    try:
        return path.resolve().relative_to(settings.benchmarks_dir.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


# ---------------------------------------------------------------------------
# Benchmark Manifest Model (PA-004)
# ---------------------------------------------------------------------------

class BenchmarkManifest(BaseModel):
    schema_version: Literal[2] = 2
    language: str = Field(..., min_length=1)
    version: int = Field(..., ge=1)
    files: dict[str, str] = Field(default_factory=dict, description="relative path -> SHA-256")
    item_count: int = Field(default=0, ge=0)
    categories: dict[str, int] = Field(default_factory=dict)
    built_at: str
    git_commit: str | None = None

    @property
    def manifest_id(self) -> str:
        return f"{self.language}-v{self.version}"


def benchmark_manifests_dir() -> Path:
    return settings.benchmarks_dir / "manifests"


def _existing_benchmark_versions(language: str, out_dir: Path) -> list[tuple[int, Path]]:
    found: list[tuple[int, Path]] = []
    if not out_dir.is_dir():
        return found
    for p in out_dir.glob("*.manifest.json"):
        m = _BENCHMARK_MANIFEST_NAME_RE.match(p.name)
        if m and m.group("lang") == language:
            found.append((int(m.group("n")), p))
    return sorted(found)


def latest_benchmark_manifest_path(language: str, out_dir: Path | None = None) -> Path | None:
    versions = _existing_benchmark_versions(language, out_dir or benchmark_manifests_dir())
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


def _current_benchmark_files(language: str) -> dict[str, str]:
    bench_dir = settings.language_benchmark_dir(language)
    if not bench_dir.is_dir():
        return {}
    files: dict[str, str] = {}
    for p in sorted(bench_dir.glob("*.jsonl")):
        if is_countable_benchmark_file(p):
            files[_rel_benchmark_path(p)] = _sha256_file(p)
    return files


def build_benchmark_manifest(language: str = "krio", out_dir: Path | None = None) -> Path:
    """
    Scan `benchmarks/<language>/`, validate every BenchmarkItem, and write a versioned manifest.
    Raises ValueError if items fail validation or have review_count < 2.
    """
    target_dir = out_dir or benchmark_manifests_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    files_map = _current_benchmark_files(language)
    item_ids: set[str] = set()
    category_counts: Counter[str] = Counter()

    for rel_path in files_map:
        full_path = settings.benchmarks_dir / rel_path
        with full_path.open(encoding="utf-8-sig") as f:
            for lineno, raw_line in enumerate(f, 1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    obj = json.loads(line)
                    meta = obj.get("metadata", {})
                    source = meta.get("source", "")
                    if _is_template_record(source):
                        continue
                    item = BenchmarkItem.model_validate(obj)
                    if item.review_count < 2:
                        raise ValueError(
                            f"Item {item.item_id} in {rel_path}:{lineno} has review_count={item.review_count} < 2"
                        )
                    item_ids.add(item.item_id)
                    category_counts[item.evaluation_category] += 1
                except Exception as exc:
                    raise ValueError(
                        f"Benchmark validation failed in {rel_path} line {lineno}: {exc}"
                    ) from exc

    versions = _existing_benchmark_versions(language, target_dir)
    next_version = (versions[-1][0] + 1) if versions else 1

    manifest = BenchmarkManifest(
        language=language,
        version=next_version,
        files=files_map,
        item_count=len(item_ids),
        categories=dict(sorted(category_counts.items())),
        built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_commit=_git_commit(),
    )

    out_path = target_dir / f"{manifest.manifest_id}.manifest.json"
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest.model_dump(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, out_path)
    return out_path


# ---------------------------------------------------------------------------
# Text normalisation and fingerprinting
# ---------------------------------------------------------------------------

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
    Template files and template records are excluded.
    """
    bench_dir = settings.language_benchmark_dir(language)
    if not bench_dir.exists():
        return set()

    fps: set[str] = set()
    for jf in bench_dir.glob("*.jsonl"):
        if is_schema_template_file(jf) or not is_countable_benchmark_file(jf):
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


def count_validated_benchmark_items(language: str = "krio") -> int:
    """
    Count held-out benchmark items that are schema-valid, VALIDATED, split=held_out
    and not templates, de-duplicated by record_id (PA-001). Malformed rows count 0.
    """
    bench_dir = settings.language_benchmark_dir(language)
    if not bench_dir.is_dir():
        return 0
    ids: set[str] = set()
    for jf in sorted(bench_dir.glob("*.jsonl")):
        if is_schema_template_file(jf) or not is_countable_benchmark_file(jf):
            continue
        with jf.open(encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    rec = json.loads(line)
                    meta = DatasetMetadata.model_validate(rec.get("metadata"))
                except Exception:
                    continue
                if (
                    _is_template_record(meta.source)
                    or meta.validation_status is not ValidationStatus.VALIDATED
                    or meta.split is not DataSplit.HELD_OUT
                ):
                    continue
                ids.add(meta.record_id)
    return len(ids)


def check_held_out_integrity(
    language: str = "krio",
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """
    Verify held-out benchmark integrity (PA-004):
    1. Directory exists
    2. Valid benchmark manifest exists
    3. Manifest file hashes match files on disk (no missing/modified files)
    4. No unmanifested real benchmark data files on disk
    5. Count of validated items >= min_benchmark_items (config; default 300)
    6. All items validate as BenchmarkItem with review_count >= 2

    Returns: {'ok': bool, 'count': int, 'reasons': list[str]}
    """
    bench_dir = settings.language_benchmark_dir(language)
    reasons: list[str] = []

    if not bench_dir.is_dir():
        return {"ok": False, "count": 0, "reasons": [f"directory_missing:{bench_dir}"]}

    path = manifest_path or latest_benchmark_manifest_path(language)
    if path is None:
        reasons.append("no_benchmark_manifest")
        return {"ok": False, "count": 0, "reasons": reasons}

    try:
        manifest_data = json.loads(path.read_text(encoding="utf-8-sig"))
        manifest = BenchmarkManifest.model_validate(manifest_data)
    except Exception as exc:
        reasons.append(f"invalid_manifest:{exc}")
        return {"ok": False, "count": 0, "reasons": reasons}

    if manifest.language != language:
        reasons.append(f"manifest_language_mismatch:{manifest.language}!={language}")

    # Check file hashes
    for rel_path, expected_hash in manifest.files.items():
        file_path = settings.benchmarks_dir / rel_path
        if not file_path.is_file():
            reasons.append(f"missing_file:{rel_path}")
        else:
            actual_hash = _sha256_file(file_path)
            if actual_hash.lower() != expected_hash.lower():
                reasons.append(f"hash_mismatch:{rel_path}")

    # Check for unmanifested data files
    disk_files = _current_benchmark_files(language)
    for rel_path in sorted(set(disk_files) - set(manifest.files)):
        reasons.append(f"unmanifested_file:{rel_path}")

    # Verify minimum count threshold
    min_needed = getattr(settings, "min_benchmark_items", 300)
    if manifest.item_count < min_needed:
        reasons.append(f"insufficient_items:{manifest.item_count}<{min_needed}")

    ok = len(reasons) == 0
    return {"ok": ok, "count": manifest.item_count, "reasons": reasons}


def assert_held_out_integrity(language: str = "krio") -> bool:
    """
    Verify that the held-out benchmark is in place and hash-verified (PA-004).
    Returns True if valid, False otherwise.
    """
    result = check_held_out_integrity(language)
    if not result["ok"]:
        logger.warning(
            f"[BENCHMARK] Held-out integrity failed for {language}: {result['reasons']}"
        )
        return False
    logger.info(f"[BENCHMARK] {language}: {result['count']} items verified.")
    return True
