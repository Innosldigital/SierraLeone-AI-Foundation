"""
Dataset loading utilities for Sierra Leonean language data.

Loads datasets from JSONL files, applies a strict eligibility gate,
and returns validated record objects wrapped in a LoadReport (which also
behaves as a sequence of records for backward compatibility).

ELIGIBILITY RULES (enforced on every load)
-------------------------------------------
* Records whose source starts with SCHEMA_TEMPLATE… / BENCHMARK_SCHEMA_TEMPLATE…
  are always excluded (they are schema examples, not real data).
* REJECTED records are never returned (rule: "rejected records must never be used for training").
* MACHINE_GENERATED records are excluded unless `policy.include_machine_generated=True`.
* Records with `consent_withdrawn=True` are always excluded.
* Records with `licence='unknown'`, `licence_id='unknown'`, or `training_permitted=False`
  are excluded from training loads (SLAI-025).
* Records whose source is NOT listed in `datasets/source_register.csv` with decision
  `APPROVED` are excluded. The register is an allow-list; a missing register fails
  closed (PA-008).
* Records are parsed into validated Pydantic models BEFORE eligibility is evaluated;
  eligibility never inspects raw dicts (PA-002). The task-specific consent scope must
  be present (e.g. `text_training` for text loads, PA-003).
* Records where `contributor_consent=False` are excluded unless `policy.require_consent=False`.
* Records whose `split` field does not match the requested split are excluded.
* Duplicate `record_id` values are excluded (SLAI-028).
* `validated_only=True` accepts only ValidationStatus.VALIDATED (≥2 native speaker reviews).
  PARTIALLY_VALIDATED (one reviewer) is excluded when validated_only is True.
* In translation loads: translation_method == "machine_only" is always excluded
  (Absolute Rule 7 from CLAUDE.md).

STRICT MODE (SLAI-022)
----------------------
* `strict=True` (default for training/release loaders): any malformed JSON line or
  schema validation failure raises `DataValidationError` carrying the `LoadReport`
  with exact line numbers and error details.
* `strict=False` (PREVIEW mode – not for training): returns valid rows alongside
  `report.issues`.
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Generic, TypeVar, overload

from loguru import logger

from sierraleone_ai.config import settings
from sierraleone_ai.datasets.constants import TEMPLATE_PREFIXES
from sierraleone_ai.datasets.eligibility import EligibilityPolicy, reasons_ineligible
from sierraleone_ai.datasets.schema import (
    AgriculturalConversationRecord,
    ConversationRecord,
    DatasetMetadata,
    KrioTextRecord,
    LicenceType,
    TranslationRecord,
    ValidationStatus,
)

T = TypeVar("T")

# Backward-compatible aliases
LoadPolicy = EligibilityPolicy
_TEMPLATE_PREFIXES = TEMPLATE_PREFIXES

# PII patterns for consent store separation lint (SLAI-025)
_EMAIL_PII_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_PII_RE = re.compile(r"(?:\+?232[\s-]?[0-9]{2}[\s-]?[0-9]{6}|\b0[0-9]{2}[\s-]?[0-9]{6}\b)")


# ---------------------------------------------------------------------------
# Load report & strict validation error (SLAI-022)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LoadIssue:
    """Line-level record of a malformed JSON line or schema validation error."""

    path: str
    line: int
    record_id: str | None
    error: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LoadReport(Generic[T]):
    """
    Result of loading a dataset file (SLAI-022).

    Exposes `.records`, `.issues`, `.skipped`, and `.total_lines`, while also
    implementing sequence operations (`len()`, iteration, indexing, list equality)
    over `.records` for seamless backward compatibility.
    """

    records: list[T] = field(default_factory=list)
    issues: list[LoadIssue] = field(default_factory=list)
    skipped: dict[str, int] = field(default_factory=dict)
    total_lines: int = 0
    preview_mode: bool = False

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[T]:
        return iter(self.records)

    @overload
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, index: slice) -> list[T]: ...

    def __getitem__(self, index: int | slice) -> T | list[T]:
        return self.records[index]

    def __bool__(self) -> bool:
        return bool(self.records)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, list):
            return self.records == other
        if isinstance(other, LoadReport):
            return (
                self.records == other.records
                and self.issues == other.issues
                and self.skipped == other.skipped
            )
        return NotImplemented


class DataValidationError(ValueError):
    """Raised in strict mode when a dataset file contains malformed JSON or invalid records."""

    def __init__(self, message: str, report: LoadReport[Any]) -> None:
        super().__init__(message)
        self.report = report


# ---------------------------------------------------------------------------
# Source rights register (SLAI-031)
# ---------------------------------------------------------------------------

def load_source_register(register_path: Path | None = None) -> dict[str, dict[str, str]]:
    """
    Load `datasets/source_register.csv` mapping `source_name` to its audit metadata.
    Sources with `decision != 'APPROVED'` (e.g. `UNREVIEWED`) are ineligible for training.
    """
    path = register_path or (settings.data_dir / "source_register.csv")
    if not path.exists():
        # Fail closed: a missing register means no source is approved (PA-008).
        return {}
    register: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("source_name") or "").strip()
            if name:
                register[name] = {k: (v or "").strip() for k, v in row.items() if k}
    return register


# ---------------------------------------------------------------------------
# Eligibility gate
# ---------------------------------------------------------------------------

def _is_eligible(
    record_or_dict: dict[str, Any] | Any,
    split: str,
    policy: EligibilityPolicy,
    source_register: dict[str, dict[str, str]] | None = None,
) -> tuple[bool, str]:
    """
    Return (eligible, skip_reason).
    skip_reason is "" when eligible.
    """
    meta: DatasetMetadata | None = None
    rec: Any = None
    meta_attr = getattr(record_or_dict, "metadata", None)
    if isinstance(record_or_dict, DatasetMetadata):
        meta = record_or_dict
    elif isinstance(meta_attr, DatasetMetadata):
        meta = meta_attr
        rec = record_or_dict
    elif isinstance(record_or_dict, dict):
        rec = record_or_dict
        meta_dict = record_or_dict.get("metadata", {})
        if not isinstance(meta_dict, dict):
            return False, "invalid_metadata"
        try:
            meta = DatasetMetadata.model_validate(meta_dict)
        except Exception:
            return False, "invalid_metadata"
    else:
        return False, "invalid_record"

    if policy.split != split:
        policy = EligibilityPolicy(
            task=policy.task,
            split=split,
            validated_only=policy.validated_only,
            allow_partially_validated=policy.allow_partially_validated,
            include_machine_generated=policy.include_machine_generated,
            require_consent=policy.require_consent,
            enforce_source_register=policy.enforce_source_register,
            exclude_benchmark_overlap=policy.exclude_benchmark_overlap,
            allow_templates=policy.allow_templates,
        )

    reasons = reasons_ineligible(meta, rec, policy, source_register)
    if reasons:
        return False, reasons[0]
    return True, ""


# ---------------------------------------------------------------------------
# JSONL reader & quarantine writer (SLAI-022)
# ---------------------------------------------------------------------------

def _write_quarantine_issues(
    source_path: Path,
    issues: list[LoadIssue],
    language: str = "krio",
) -> Path:
    """Write malformed row issues to datasets/<lang>/quarantine/<file>.issues.jsonl."""
    q_dir = settings.language_data_dir(language) / "quarantine"
    q_dir.mkdir(parents=True, exist_ok=True)
    out_path = q_dir / f"{source_path.stem}.issues.jsonl"
    with out_path.open("w", encoding="utf-8-sig") as f:
        for issue in issues:
            f.write(json.dumps(issue.to_dict()) + "\n")
    return out_path


def _load_jsonl_with_issues(
    path: Path,
) -> tuple[list[tuple[int, dict[str, Any]]], list[LoadIssue], int]:
    """
    Read a JSONL file and return (valid_line_tuples, issues, total_nonblank_lines).
    Never logs raw line text to avoid leaking sensitive content (SLAI-028).
    """
    rows: list[tuple[int, dict[str, Any]]] = []
    issues: list[LoadIssue] = []
    total_lines = 0
    with path.open(encoding="utf-8-sig") as f:
        for lineno, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            total_lines += 1
            try:
                parsed = json.loads(line)
                if not isinstance(parsed, dict):
                    issues.append(
                        LoadIssue(
                            path=str(path),
                            line=lineno,
                            record_id=None,
                            error="JSON line is not an object",
                        )
                    )
                    continue
                rows.append((lineno, parsed))
            except json.JSONDecodeError as exc:
                logger.warning(f"{path.name}:{lineno} – JSON decode error at pos {exc.pos}")
                issues.append(
                    LoadIssue(
                        path=str(path),
                        line=lineno,
                        record_id=None,
                        error=f"JSONDecodeError: {exc.msg} (line {lineno}, col {exc.colno})",
                    )
                )
    return rows, issues, total_lines


def _load_jsonl(path: Path, *, strict: bool = False) -> list[dict[str, Any]]:
    """
    Load a JSONL file, skipping blank lines and comment lines.
    If strict=True, raises DataValidationError on any malformed JSON line.
    """
    rows, issues, total_lines = _load_jsonl_with_issues(path)
    records = [r for _, r in rows]
    if strict and issues:
        report: LoadReport[dict[str, Any]] = LoadReport(
            records=records,
            issues=issues,
            skipped={},
            total_lines=total_lines,
        )
        first = issues[0]
        raise DataValidationError(
            f"Malformed JSON in {path.name} at line {first.line}: {first.error}",
            report=report,
        )
    return records


def _apply_policy(
    raw_records: list[dict[str, Any] | Any],
    split: str,
    policy: EligibilityPolicy,
    file_name: str,
    *,
    seen_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any] | Any], Counter[str]]:
    """Filter raw dicts or models through the eligibility gate; return admitted records + skip counter."""
    admitted: list[dict[str, Any] | Any] = []
    skipped: Counter[str] = Counter()
    source_reg = load_source_register() if policy.enforce_source_register else None
    tracked_ids = seen_ids if seen_ids is not None else set()

    if policy.split != split:
        policy = EligibilityPolicy(
            task=policy.task,
            split=split,
            validated_only=policy.validated_only,
            allow_partially_validated=policy.allow_partially_validated,
            include_machine_generated=policy.include_machine_generated,
            require_consent=policy.require_consent,
            enforce_source_register=policy.enforce_source_register,
            exclude_benchmark_overlap=policy.exclude_benchmark_overlap,
            allow_templates=policy.allow_templates,
        )

    for rec in raw_records:
        meta: DatasetMetadata | None = None
        if isinstance(rec, dict):
            meta_dict = rec.get("metadata", {})
            if not isinstance(meta_dict, dict):
                skipped["invalid_metadata"] += 1
                continue
            try:
                meta = DatasetMetadata.model_validate(meta_dict)
            except Exception:
                skipped["invalid_metadata"] += 1
                continue
        else:
            rec_meta = getattr(rec, "metadata", None)
            if isinstance(rec_meta, DatasetMetadata):
                meta = rec_meta

        if meta is None:
            skipped["invalid_record"] += 1
            continue

        reasons = reasons_ineligible(meta, rec, policy, source_reg)
        if reasons:
            for r in reasons:
                skipped[r] += 1
            continue

        rec_id = str(meta.record_id) if meta.record_id else None
        if rec_id:
            if rec_id in tracked_ids:
                skipped["duplicate_record_id"] += 1
                continue
            tracked_ids.add(rec_id)
        admitted.append(rec)

    total = len(raw_records)
    n_skip = sum(skipped.values())
    if n_skip:
        logger.info(
            f"{file_name}: {len(admitted)}/{total} records admitted; "
            f"skipped: {dict(skipped)}"
        )
    return admitted, skipped


# ---------------------------------------------------------------------------
# PII scanner & release export filter (SLAI-025)
# ---------------------------------------------------------------------------

def scan_jsonl_for_pii(path: Path) -> list[LoadIssue]:
    """
    Scan a JSONL file for email addresses or phone numbers (SLAI-025).
    Identifiable consent details must live in a separate restricted store, never in JSONL.
    """
    issues: list[LoadIssue] = []
    if not path.exists():
        return issues
    with path.open(encoding="utf-8-sig") as f:
        for lineno, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            rec_id: str | None = None
            try:
                obj = json.loads(stripped)
                if isinstance(obj, dict):
                    rec_id = obj.get("metadata", {}).get("record_id")
            except json.JSONDecodeError:
                pass
            if _EMAIL_PII_RE.search(stripped):
                issues.append(
                    LoadIssue(
                        path=str(path),
                        line=lineno,
                        record_id=rec_id,
                        error="PII detected: email address pattern in dataset JSONL",
                    )
                )
            if _PHONE_PII_RE.search(stripped):
                issues.append(
                    LoadIssue(
                        path=str(path),
                        line=lineno,
                        record_id=rec_id,
                        error="PII detected: phone number pattern in dataset JSONL",
                    )
                )
    return issues


def export_for_release(records: Iterable[T]) -> list[T]:
    """
    Filter validated records eligible for public release export (SLAI-025).

    Excludes records where:
      - licence is unknown or redistribution_permitted is False
      - access_class != 'public'
      - contributor_consent is False or consent_withdrawn is True
      - neither 'public_release' nor 'publication' is in consent_scope
    """
    released: list[T] = []
    for rec in records:
        meta = getattr(rec, "metadata", None)
        if meta is None:
            continue
        if meta.licence == LicenceType.UNKNOWN or meta.licence_id == "unknown":
            continue
        if not meta.redistribution_permitted:
            continue
        if meta.access_class != "public":
            continue
        if not meta.contributor_consent or meta.consent_withdrawn:
            continue
        if not any(s in ("public_release", "publication") for s in meta.consent_scope):
            continue
        released.append(rec)
    return released


# ---------------------------------------------------------------------------
# Public loaders (SLAI-001 / SLAI-007 / SLAI-022 / PA-002 / PA-009)
# ---------------------------------------------------------------------------

def load_jsonl_records(
    path: Path,
    model_cls: type[T],
    split: str = "train",
    policy: EligibilityPolicy | None = None,
    *,
    strict: bool = True,
    language: str = "krio",
    write_quarantine: bool = False,
) -> LoadReport[T]:
    """
    Generic strict/preview JSONL loader with line-level error reporting (SLAI-022 / PA-002 / PA-009).
    """
    if policy is None:
        policy = EligibilityPolicy(split=split)
    elif policy.split != split:
        policy = EligibilityPolicy(
            task=policy.task,
            split=split,
            validated_only=policy.validated_only,
            allow_partially_validated=policy.allow_partially_validated,
            include_machine_generated=policy.include_machine_generated,
            require_consent=policy.require_consent,
            enforce_source_register=policy.enforce_source_register,
            exclude_benchmark_overlap=policy.exclude_benchmark_overlap,
            allow_templates=policy.allow_templates,
        )

    if not path.exists():
        logger.warning(f"Dataset file not found: {path}")
        return LoadReport(preview_mode=not strict)

    if not strict:
        logger.info(f"PREVIEW – not for training: loading {path.name} with strict=False")

    rows, issues, total_lines = _load_jsonl_with_issues(path)
    source_reg = load_source_register() if policy.enforce_source_register else None
    skipped: Counter[str] = Counter()
    seen_ids: set[str] = set()

    bench_fps: set[str] = set()
    if policy.exclude_benchmark_overlap:
        from sierraleone_ai.evaluation.benchmarks import benchmark_fingerprints
        bench_fps = benchmark_fingerprints(language)

    records: list[T] = []
    for lineno, item in rows:
        meta_raw = item.get("metadata")
        if not isinstance(meta_raw, dict):
            issues.append(
                LoadIssue(
                    path=str(path),
                    line=lineno,
                    record_id=None,
                    error=f"Record metadata must be a JSON object, got {type(meta_raw).__name__}",
                )
            )
            continue

        rec_id_raw = meta_raw.get("record_id")
        rec_id_str = str(rec_id_raw) if rec_id_raw is not None else None

        # 1. Parse into model first (PA-002 / PA-009a)
        try:
            if model_cls is ConversationRecord:
                is_agri = (
                    item.get("domain") == "agriculture"
                    or "crop_or_livestock" in item
                    or "agro_ecological_zone" in item
                )
                parsed_obj: Any = (
                    AgriculturalConversationRecord(**item)
                    if is_agri
                    else ConversationRecord(**item)
                )
            else:
                parsed_obj = model_cls(**item)
        except Exception as exc:
            logger.warning(f"Schema validation failed in {path.name}:{lineno} (id={rec_id_str})")
            issues.append(
                LoadIssue(
                    path=str(path),
                    line=lineno,
                    record_id=rec_id_str,
                    error=str(exc),
                )
            )
            continue

        # 2. Evaluate eligibility on validated model (PA-002 / PA-003 / PA-008)
        parsed_meta = getattr(parsed_obj, "metadata", None)
        if not isinstance(parsed_meta, DatasetMetadata):
            issues.append(
                LoadIssue(
                    path=str(path),
                    line=lineno,
                    record_id=rec_id_str,
                    error="Parsed record is missing valid DatasetMetadata",
                )
            )
            continue
        reasons = reasons_ineligible(parsed_meta, parsed_obj, policy, source_reg)
        if reasons:
            for r in reasons:
                skipped[r] += 1
            continue

        # 3. Duplicate ID check
        if rec_id_str:
            if rec_id_str in seen_ids:
                skipped["duplicate_record_id"] += 1
                continue
            seen_ids.add(rec_id_str)

        # 4. Held-out benchmark overlap check on all text fields (PA-004)
        if bench_fps:
            from sierraleone_ai.evaluation.benchmarks import iter_texts, text_fingerprint
            if any(text_fingerprint(t) in bench_fps for t in iter_texts(parsed_obj)):
                skipped["benchmark_overlap"] += 1
                continue

        records.append(parsed_obj)

    if issues and write_quarantine:
        _write_quarantine_issues(path, issues, language=language)

    report = LoadReport(
        records=records,
        issues=issues,
        skipped=dict(skipped),
        total_lines=total_lines,
        preview_mode=not strict,
    )

    if strict and issues:
        first = issues[0]
        raise DataValidationError(
            f"Validation failed in {path.name} at line {first.line} "
            f"(record_id={first.record_id}): {first.error}",
            report=report,
        )

    return report


def load_records(
    language: str = "krio",
    kind: str = "text",
    split: str = "train",
    policy: EligibilityPolicy | None = None,
    *,
    strict: bool = True,
    write_quarantine: bool = False,
    **kwargs: Any,
) -> LoadReport[Any]:
    """
    Generic record loader across languages, splits, and record kinds (PA-002 / PA-021).
    Supports kind in ('text', 'translations', 'conversations').
    """
    lang_dir = settings.language_data_dir(language)
    if kind == "text":
        path = lang_dir / "cleaned" / f"{language}_text_{split}.jsonl"
        if not path.exists():
            val_path = lang_dir / "validated" / f"{language}_text_{split}.jsonl"
            if val_path.exists():
                path = val_path
        if policy is None:
            policy = EligibilityPolicy(task="text_training", split=split)
        return load_jsonl_records(
            path,
            KrioTextRecord,
            split=split,
            policy=policy,
            strict=strict,
            language=language,
            write_quarantine=write_quarantine,
        )
    elif kind in ("translation", "translations"):
        source_lang = kwargs.get("source_lang", "english")
        target_lang = kwargs.get("target_lang", language)
        fname = f"translations_{source_lang}_{target_lang}_{split}.jsonl"
        path = settings.language_data_dir(target_lang) / "translations" / fname
        if policy is None:
            policy = EligibilityPolicy(task="translation_training", split=split)
        return load_jsonl_records(
            path,
            TranslationRecord,
            split=split,
            policy=policy,
            strict=strict,
            language=target_lang,
            write_quarantine=write_quarantine,
        )
    elif kind in ("conversation", "conversations"):
        path = lang_dir / "conversations" / f"conversations_{split}.jsonl"
        if policy is None:
            policy = EligibilityPolicy(task="conversation_training", split=split)
        report = load_jsonl_records(
            path,
            ConversationRecord,
            split=split,
            policy=policy,
            strict=strict,
            language=language,
            write_quarantine=write_quarantine,
        )
        domain = kwargs.get("domain")
        if domain:
            report.records = [r for r in report.records if getattr(r, "domain", None) == domain]
        return report
    else:
        raise ValueError(f"Unknown record kind: {kind}")


def load_krio_text(
    split: str = "train",
    policy: EligibilityPolicy | None = None,
    *,
    strict: bool = True,
    write_quarantine: bool = False,
) -> LoadReport[KrioTextRecord]:
    """
    Load Krio text records for a given split.
    Defaults to strict=True (raises DataValidationError on malformed lines).
    """
    return load_records(
        language="krio",
        kind="text",
        split=split,
        policy=policy,
        strict=strict,
        write_quarantine=write_quarantine,
    )


def load_translations(
    source_lang: str = "english",
    target_lang: str = "krio",
    split: str = "train",
    policy: EligibilityPolicy | None = None,
    *,
    strict: bool = True,
    write_quarantine: bool = False,
) -> LoadReport[TranslationRecord]:
    """
    Load parallel translation records.
    machine_only translations are always excluded regardless of policy (Rule 7).
    """
    return load_records(
        language=target_lang,
        kind="translations",
        split=split,
        policy=policy,
        strict=strict,
        write_quarantine=write_quarantine,
        source_lang=source_lang,
        target_lang=target_lang,
    )


def load_conversations(
    language: str = "krio",
    split: str = "train",
    domain: str | None = None,
    policy: EligibilityPolicy | None = None,
    *,
    strict: bool = True,
    write_quarantine: bool = False,
) -> LoadReport[ConversationRecord]:
    """
    Load conversation records, optionally filtered by domain.
    Agricultural conversations are parsed as AgriculturalConversationRecord.
    """
    return load_records(
        language=language,
        kind="conversations",
        split=split,
        policy=policy,
        strict=strict,
        write_quarantine=write_quarantine,
        domain=domain,
    )


# ---------------------------------------------------------------------------
# Dataset summary – PREVIEW only (PA-001 / PA-007)
# ---------------------------------------------------------------------------
# Training readiness is NOT computed here; it comes from the hash-verified dataset
# manifest (see datasets/manifest.py and training/guard.py::check_dataset_readiness).

def _count_schema_valid_records(path: Path) -> int:
    """
    Count rows whose metadata is schema-valid, not a template, and not REJECTED.
    Rows without valid metadata (including quarantine issue rows) count 0 (PA-007).
    """
    count = 0
    rows, _issues, _total = _load_jsonl_with_issues(path)
    for _lineno, rec in rows:
        meta_raw = rec.get("metadata")
        if not isinstance(meta_raw, dict):
            continue
        try:
            meta = DatasetMetadata.model_validate(meta_raw)
        except Exception:
            continue
        if meta.source.upper().startswith(TEMPLATE_PREFIXES):
            continue
        if meta.validation_status is ValidationStatus.REJECTED:
            continue
        count += 1
    return count


def dataset_summary() -> dict[str, dict[str, int]]:
    """
    Return a PREVIEW summary: language → subdirectory → schema-valid record count.

    Skips templates (`example_*`), quarantine reports (`*.issues.jsonl`), and the
    `quarantine/`, `splits/`, `manifests/`, `raw/`, `rejected/` folders. These numbers
    are NOT training-eligible counts – use `slai build-manifest` / `slai check-readiness`.
    Returns {} if the data directory does not exist.
    """
    from sierraleone_ai.datasets.manifest import is_countable_data_file

    if not settings.data_dir.exists():
        return {}

    summary: dict[str, dict[str, int]] = {}
    for lang_dir in sorted(settings.data_dir.iterdir()):
        if not lang_dir.is_dir() or lang_dir.name == "manifests":
            continue
        lang = lang_dir.name
        summary[lang] = {}
        for sub in sorted(lang_dir.iterdir()):
            if not sub.is_dir():
                continue
            total = 0
            for jf in sorted(sub.glob("*.jsonl")):
                if is_countable_data_file(jf):
                    total += _count_schema_valid_records(jf)
            if total > 0:
                summary[lang][sub.name] = total
    return summary
