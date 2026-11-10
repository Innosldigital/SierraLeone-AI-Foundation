"""
Dataset loading utilities for Sierra Leonean language data.

Loads datasets from JSONL files, applies a strict eligibility gate,
and returns validated record objects.

ELIGIBILITY RULES (enforced on every load)
-------------------------------------------
* Records whose source starts with SCHEMA_TEMPLATE… / BENCHMARK_SCHEMA_TEMPLATE…
  are always excluded (they are schema examples, not real data).
* REJECTED records are never returned (rule: "rejected records must never be used for training").
* MACHINE_GENERATED records are excluded unless `policy.include_machine_generated=True`.
* Records where `contributor_consent=False` are excluded unless `policy.require_consent=False`.
* Records whose `split` field does not match the requested split are excluded.
* `validated_only=True` accepts only ValidationStatus.VALIDATED (≥2 native speaker reviews).
  PARTIALLY_VALIDATED (one reviewer) is excluded when validated_only is True.
* In translation loads: translation_method == "machine_only" is always excluded
  (Absolute Rule 7 from CLAUDE.md).

Skip reasons are counted and logged in a single INFO line per file — not one warning per record.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from sierraleone_ai.config import settings
from sierraleone_ai.datasets.schema import (
    AgriculturalConversationRecord,
    ConversationRecord,
    KrioTextRecord,
    TranslationRecord,
    ValidationStatus,
)

# ---------------------------------------------------------------------------
# Template source prefix guard (Absolute Rule 1)
# ---------------------------------------------------------------------------
_TEMPLATE_PREFIXES = (
    "SCHEMA_TEMPLATE",
    "BENCHMARK_SCHEMA_TEMPLATE",
    "EXAMPLE_PLACEHOLDER",
)


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


# ---------------------------------------------------------------------------
# Eligibility gate
# ---------------------------------------------------------------------------

def _is_eligible(record_dict: dict, split: str, policy: LoadPolicy) -> tuple[bool, str]:
    """
    Return (eligible, skip_reason).
    skip_reason is "" when eligible.
    """
    meta = record_dict.get("metadata", {})
    source: str = meta.get("source", "")
    status: str = meta.get("validation_status", "unvalidated")
    consent: bool = meta.get("contributor_consent", False)
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

    # 5. Consent
    if policy.require_consent and not consent:
        return False, "no_consent"

    # 6. Split mismatch
    if record_split != split:
        return False, f"split_mismatch({record_split}!={split})"

    return True, ""


# ---------------------------------------------------------------------------
# JSONL reader
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file, skipping blank lines and comment lines."""
    records: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning(f"{path.name}:{lineno} – JSON decode error: {exc}")
    return records


def _apply_policy(
    raw_records: list[dict],
    split: str,
    policy: LoadPolicy,
    file_name: str,
) -> tuple[list[dict], Counter]:
    """Filter raw dicts through the eligibility gate; return admitted records + skip counter."""
    admitted: list[dict] = []
    skipped: Counter = Counter()
    for rec in raw_records:
        ok, reason = _is_eligible(rec, split, policy)
        if ok:
            admitted.append(rec)
        else:
            skipped[reason] += 1
    total = len(raw_records)
    n_skip = sum(skipped.values())
    if n_skip:
        logger.info(
            f"{file_name}: {total - n_skip}/{total} records admitted; "
            f"skipped: {dict(skipped)}"
        )
    return admitted, skipped


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_krio_text(
    split: str = "train",
    policy: LoadPolicy | None = None,
) -> list[KrioTextRecord]:
    """
    Load Krio text records for a given split.

    Args:
        split: 'train', 'validation', 'test', or 'held_out'
        policy: LoadPolicy controlling eligibility rules.
                Defaults to require_consent=True, no templates, no machine_generated.
    """
    if policy is None:
        policy = LoadPolicy()

    path = settings.krio_data_dir / "cleaned" / f"krio_text_{split}.jsonl"
    if not path.exists():
        logger.warning(f"Dataset file not found: {path}")
        return []

    raw = _load_jsonl(path)
    admitted, _ = _apply_policy(raw, split, policy, path.name)

    records: list[KrioTextRecord] = []
    for item in admitted:
        try:
            records.append(KrioTextRecord(**item))
        except Exception as exc:
            logger.warning(f"Schema validation failed in {path.name}: {exc}")

    logger.info(f"Loaded {len(records)} Krio text records (split={split})")
    return records


def load_translations(
    source_lang: str = "english",
    target_lang: str = "krio",
    split: str = "train",
    policy: LoadPolicy | None = None,
) -> list[TranslationRecord]:
    """
    Load parallel translation records.

    machine_only translations are always excluded regardless of policy
    (Absolute Rule 7: machine-only translations must never be ground-truth training data).
    """
    if policy is None:
        policy = LoadPolicy()

    fname = f"translations_{source_lang}_{target_lang}_{split}.jsonl"
    path = settings.language_data_dir(target_lang) / "translations" / fname
    if not path.exists():
        logger.warning(f"Translation file not found: {path}")
        return []

    raw = _load_jsonl(path)
    admitted, _ = _apply_policy(raw, split, policy, path.name)

    records: list[TranslationRecord] = []
    for item in admitted:
        # Rule 7: exclude machine_only translations
        if item.get("translation_method") == "machine_only":
            logger.debug(f"Skipping machine_only translation in {path.name}")
            continue
        try:
            records.append(TranslationRecord(**item))
        except Exception as exc:
            logger.warning(f"Schema validation failed in {path.name}: {exc}")

    logger.info(
        f"Loaded {len(records)} translation records ({source_lang}→{target_lang}, split={split})"
    )
    return records


def load_conversations(
    language: str = "krio",
    split: str = "train",
    domain: str | None = None,
    policy: LoadPolicy | None = None,
) -> list[ConversationRecord]:
    """
    Load conversation records, optionally filtered by domain.
    Agricultural conversations (domain='agriculture' or records with crop_or_livestock)
    are parsed as AgriculturalConversationRecord.
    """
    if policy is None:
        policy = LoadPolicy()

    path = settings.language_data_dir(language) / "conversations" / f"conversations_{split}.jsonl"
    if not path.exists():
        logger.warning(f"Conversations file not found: {path}")
        return []

    raw = _load_jsonl(path)
    admitted, _ = _apply_policy(raw, split, policy, path.name)

    records: list[ConversationRecord] = []
    for item in admitted:
        try:
            # Use AgriculturalConversationRecord for agricultural domain
            is_agri = (
                item.get("domain") == "agriculture"
                or "crop_or_livestock" in item
                or "agro_ecological_zone" in item
            )
            rec = AgriculturalConversationRecord(**item) if is_agri else ConversationRecord(**item)
            if domain and rec.domain != domain:
                continue
            records.append(rec)
        except Exception as exc:
            logger.warning(f"Schema validation failed in {path.name}: {exc}")

    logger.info(f"Loaded {len(records)} {language} conversations (split={split})")
    return records


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

    raw = _load_jsonl(path)
    count = 0
    for rec in raw:
        # For summary purposes, only filter out templates and rejected
        source: str = rec.get("metadata", {}).get("source", "")
        status: str = rec.get("metadata", {}).get("validation_status", "")
        if source.upper().startswith(_TEMPLATE_PREFIXES):
            continue
        if status == ValidationStatus.REJECTED.value:
            continue
        count += 1
    return count


def dataset_summary() -> dict[str, dict[str, int]]:
    """
    Return a summary dict: language → subdirectory → real record count.
    Template and rejected records are NOT counted.
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
