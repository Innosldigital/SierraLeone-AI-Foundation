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
* Records whose source is listed in `datasets/source_register.csv` with a decision other
  than `APPROVED` (e.g. `UNREVIEWED` or `REJECTED`) are excluded (SLAI-031).
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
from sierraleone_ai.datasets.schema import (
    AgriculturalConversationRecord,
    ConversationRecord,
    KrioTextRecord,
    LicenceType,
    TranslationRecord,
    ValidationStatus,
)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Template source prefix guard (Absolute Rule 1)
# ---------------------------------------------------------------------------
_TEMPLATE_PREFIXES = (
    "SCHEMA_TEMPLATE",
    "BENCHMARK_SCHEMA_TEMPLATE",
    "EXAMPLE_PLACEHOLDER",
)

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
        return {}
    register: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("source_name") or "").strip()
            if name:
                register[name] = {k: (v or "").strip() for k, v in row.items() if k}
    return register


# ---------------------------------------------------------------------------
# Load policy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LoadPolicy:
    """Controls which records are admitted during a dataset load."""

    validated_only: bool = False
    """True → only ValidationStatus.VALIDATED records (requires ≥2 native speaker reviews)."""

    include_machine_generated: bool = False
    """Allow machine_generated records (never use for supervised training ground truth)."""

    require_consent: bool = True
    """Exclude records where contributor_consent is False."""

    allow_templates: bool = False
    """Allow template/example records. Only for unit tests."""

    exclude_benchmark_overlap: bool = True
    """Exclude records whose text fingerprint appears in the held-out benchmark."""

    enforce_source_register: bool = True
    """Exclude sources listed as UNREVIEWED or REJECTED in datasets/source_register.csv."""


# ---------------------------------------------------------------------------
# Eligibility gate
# ---------------------------------------------------------------------------

def _is_eligible(
    record_dict: dict[str, Any],
    split: str,
    policy: LoadPolicy,
    source_register: dict[str, dict[str, str]] | None = None,
) -> tuple[bool, str]:
    """
    Return (eligible, skip_reason).
    skip_reason is "" when eligible.
    """
    meta = record_dict.get("metadata", {})
    source: str = meta.get("source", "")
    status: str = meta.get("validation_status", "unvalidated")
    consent: bool = meta.get("contributor_consent", False)
    consent_withdrawn: bool = meta.get("consent_withdrawn", False)
    licence: str = meta.get("licence", "CC-BY-4.0")
    licence_id: str = meta.get("licence_id") or licence
    training_permitted: bool = meta.get("training_permitted", True)
    record_split: str = meta.get("split", "train")

    # 1. Template guard
    if not policy.allow_templates:
        if source.upper().startswith(_TEMPLATE_PREFIXES):
            return False, "template_record"

    # 2. Never admit rejected records
    if status == ValidationStatus.REJECTED.value:
        return False, "rejected"

    # 3. Machine generated
    if status == ValidationStatus.MACHINE_GENERATED.value and not policy.include_machine_generated:
        return False, "machine_generated"

    # 4. Validated-only mode
    if policy.validated_only and status != ValidationStatus.VALIDATED.value:
        return False, f"not_validated({status})"

    # 5. Source rights register check (SLAI-031)
    if policy.enforce_source_register and source_register and source in source_register:
        decision = source_register[source].get("decision", "UNREVIEWED").upper()
        if decision != "APPROVED":
            return False, f"unreviewed_source_rights({decision})"

    # 6. Unknown licence or training_permitted=False (SLAI-025)
    if licence == LicenceType.UNKNOWN.value or licence_id == "unknown" or not training_permitted:
        return False, "unknown_or_unpermitted_rights"

    # 7. Consent withdrawal & consent check
    if consent_withdrawn:
        return False, "consent_withdrawn"
    if policy.require_consent and not consent:
        return False, "no_consent"

    # 8. Split mismatch
    if record_split != split:
        return False, f"split_mismatch({record_split}!={split})"

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
    with out_path.open("w", encoding="utf-8") as f:
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
    with path.open(encoding="utf-8") as f:
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
    raw_records: list[dict[str, Any]],
    split: str,
    policy: LoadPolicy,
    file_name: str,
    *,
    seen_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Filter raw dicts through the eligibility gate; return admitted records + skip counter."""
    admitted: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    source_reg = load_source_register() if policy.enforce_source_register else None
    tracked_ids = seen_ids if seen_ids is not None else set()

    for rec in raw_records:
        ok, reason = _is_eligible(rec, split, policy, source_reg)
        if not ok:
            skipped[reason] += 1
            continue
        rec_id = rec.get("metadata", {}).get("record_id")
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
            f"{file_name}: {total - n_skip}/{total} records admitted; "
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
    with path.open(encoding="utf-8") as f:
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
# Public loaders (SLAI-001 / SLAI-007 / SLAI-022)
# ---------------------------------------------------------------------------

def load_jsonl_records(
    path: Path,
    model_cls: type[T],
    split: str = "train",
    policy: LoadPolicy | None = None,
    *,
    strict: bool = True,
    language: str = "krio",
    write_quarantine: bool = False,
) -> LoadReport[T]:
    """
    Generic strict/preview JSONL loader with line-level error reporting (SLAI-022).
    """
    if policy is None:
        policy = LoadPolicy()

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
        rec_id = item.get("metadata", {}).get("record_id") if isinstance(item.get("metadata"), dict) else None
        ok, reason = _is_eligible(item, split, policy, source_reg)
        if not ok:
            skipped[reason] += 1
            continue
        if rec_id:
            if rec_id in seen_ids:
                skipped["duplicate_record_id"] += 1
                continue
            seen_ids.add(rec_id)

        # Held-out benchmark overlap check (SLAI-007)
        if bench_fps:
            from sierraleone_ai.evaluation.benchmarks import text_fingerprint
            candidate_texts = [
                item.get("text"),
                item.get("source_text"),
                item.get("target_text"),
            ]
            if any(isinstance(t, str) and text_fingerprint(t) in bench_fps for t in candidate_texts):
                skipped["benchmark_overlap"] += 1
                continue

        # Rule 7 for TranslationRecord: exclude machine_only
        if model_cls is TranslationRecord and item.get("translation_method") == "machine_only":
            skipped["machine_only_translation"] += 1
            continue

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
            records.append(parsed_obj)
        except Exception as exc:
            logger.warning(f"Schema validation failed in {path.name}:{lineno} (id={rec_id})")
            issues.append(
                LoadIssue(
                    path=str(path),
                    line=lineno,
                    record_id=rec_id,
                    error=str(exc),
                )
            )

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


def load_krio_text(
    split: str = "train",
    policy: LoadPolicy | None = None,
    *,
    strict: bool = True,
    write_quarantine: bool = False,
) -> LoadReport[KrioTextRecord]:
    """
    Load Krio text records for a given split.
    Defaults to strict=True (raises DataValidationError on malformed lines).
    """
    path = settings.krio_data_dir / "cleaned" / f"krio_text_{split}.jsonl"
    return load_jsonl_records(
        path,
        KrioTextRecord,
        split=split,
        policy=policy,
        strict=strict,
        language="krio",
        write_quarantine=write_quarantine,
    )


def load_translations(
    source_lang: str = "english",
    target_lang: str = "krio",
    split: str = "train",
    policy: LoadPolicy | None = None,
    *,
    strict: bool = True,
    write_quarantine: bool = False,
) -> LoadReport[TranslationRecord]:
    """
    Load parallel translation records.
    machine_only translations are always excluded regardless of policy (Rule 7).
    """
    fname = f"translations_{source_lang}_{target_lang}_{split}.jsonl"
    path = settings.language_data_dir(target_lang) / "translations" / fname
    return load_jsonl_records(
        path,
        TranslationRecord,
        split=split,
        policy=policy,
        strict=strict,
        language=target_lang,
        write_quarantine=write_quarantine,
    )


def load_conversations(
    language: str = "krio",
    split: str = "train",
    domain: str | None = None,
    policy: LoadPolicy | None = None,
    *,
    strict: bool = True,
    write_quarantine: bool = False,
) -> LoadReport[ConversationRecord]:
    """
    Load conversation records, optionally filtered by domain.
    Agricultural conversations are parsed as AgriculturalConversationRecord.
    """
    path = settings.language_data_dir(language) / "conversations" / f"conversations_{split}.jsonl"
    report = load_jsonl_records(
        path,
        ConversationRecord,
        split=split,
        policy=policy,
        strict=strict,
        language=language,
        write_quarantine=write_quarantine,
    )
    if domain:
        report.records = [r for r in report.records if r.domain == domain]
    return report


# ---------------------------------------------------------------------------
# Dataset summary (SLAI-002: template files are not counted as real data)
# ---------------------------------------------------------------------------

def _count_real_records(path: Path, policy: LoadPolicy | None = None) -> int:
    """
    Count records in a JSONL file that pass the eligibility gate.
    Uses a permissive split so records from any split are counted (summary mode).
    """
    if policy is None:
        policy = LoadPolicy(require_consent=False)

    if not path.exists():
        return 0

    raw = _load_jsonl(path, strict=False)
    source_reg = load_source_register()
    count = 0
    for rec in raw:
        meta = rec.get("metadata", {})
        source: str = meta.get("source", "")
        status: str = meta.get("validation_status", "")
        if source.upper().startswith(_TEMPLATE_PREFIXES):
            continue
        if status == ValidationStatus.REJECTED.value:
            continue
        if source in source_reg and source_reg[source].get("decision", "UNREVIEWED").upper() != "APPROVED":
            continue
        count += 1
    return count


def dataset_summary() -> dict[str, dict[str, int]]:
    """
    Return a summary dict: language → subdirectory → real record count.
    Template, rejected, and unreviewed-source records are NOT counted.
    Returns {} if the data directory does not exist.
    """
    if not settings.data_dir.exists():
        return {}

    summary: dict[str, dict[str, int]] = {}
    for lang_dir in sorted(settings.data_dir.iterdir()):
        if not lang_dir.is_dir():
            continue
        lang = lang_dir.name
        summary[lang] = {}
        for sub in sorted(lang_dir.iterdir()):
            if not sub.is_dir():
                continue
            total = 0
            for jf in sub.glob("*.jsonl"):
                # Skip example/template files from the count
                if jf.stem.startswith("example_"):
                    continue
                total += _count_real_records(jf)
            if total > 0:
                summary[lang][sub.name] = total
    return summary
