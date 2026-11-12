"""
Dataset eligibility evaluation (PA-002 / PA-003 / PA-008).

Applies a single, fail-closed eligibility policy over validated Pydantic models.
Never evaluates raw dicts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from sierraleone_ai.datasets.constants import TEMPLATE_PREFIXES
from sierraleone_ai.datasets.schema import DatasetMetadata, ValidationStatus

Task = Literal[
    "text_training",
    "translation_training",
    "conversation_training",
    "asr_training",
    "tts_training",
    "evaluation",
    "public_release",
]

_REQUIRED_SCOPE: dict[str, str] = {
    "text_training": "text_training",
    "translation_training": "text_training",
    "conversation_training": "text_training",
    "asr_training": "speech_recognition",
    "tts_training": "voice_synthesis",
    "public_release": "public_release",
}


@dataclass(frozen=True)
class EligibilityPolicy:
    """Controls which records are admitted for a given task and split."""

    task: Task = "text_training"
    split: str = "train"
    validated_only: bool = True            # default to safe choice
    allow_partially_validated: bool = False
    include_machine_generated: bool = False
    require_consent: bool = True
    enforce_source_register: bool = True   # allow-list (PA-008)
    exclude_benchmark_overlap: bool = True
    allow_templates: bool = False          # tests only


def reasons_ineligible(
    meta: DatasetMetadata,
    record: Any,
    policy: EligibilityPolicy,
    register: dict[str, dict[str, str]] | None = None,
) -> list[str]:
    """
    Return every reason the record is ineligible (empty list == eligible).
    `meta` is a validated DatasetMetadata.
    """
    r: list[str] = []

    # 1. Template records
    if not policy.allow_templates and meta.source.upper().startswith(TEMPLATE_PREFIXES):
        r.append("template_record")

    # 2. Rejected status
    if meta.validation_status is ValidationStatus.REJECTED:
        r.append("rejected")

    # 3. Machine generated status
    if meta.validation_status is ValidationStatus.MACHINE_GENERATED and not policy.include_machine_generated:
        r.append("machine_generated")

    # 4. Validated status
    if policy.validated_only:
        ok = meta.validation_status is ValidationStatus.VALIDATED or (
            policy.allow_partially_validated
            and meta.validation_status is ValidationStatus.PARTIALLY_VALIDATED
        )
        if not ok:
            r.append(f"not_validated({meta.validation_status.value})")

    # 5. Rights clearance & training permission
    if meta.licence_id in (None, "unknown") or not meta.training_permitted:
        r.append("rights_not_cleared")

    # 6. Consent withdrawal
    if meta.consent_withdrawn:
        r.append("consent_withdrawn")

    # 7. Consent and consent scope
    if policy.require_consent:
        if not meta.contributor_consent:
            r.append("no_consent")
        need = _REQUIRED_SCOPE.get(policy.task)
        if need:
            if need not in meta.consent_scope:
                r.append(f"consent_scope_missing({need})")

    # 8. Split match
    if meta.split.value != policy.split:
        r.append(f"split_mismatch({meta.split.value}!={policy.split})")

    # 9. Source register allow-list (PA-008). A missing register (None) fails closed.
    if policy.enforce_source_register and not meta.source.upper().startswith(TEMPLATE_PREFIXES):
        row = (register or {}).get(meta.source)
        if row is None:
            r.append("source_not_in_register")
        elif row.get("decision", "UNREVIEWED").upper() != "APPROVED":
            r.append(f"source_rights_not_approved({row.get('decision', 'UNREVIEWED')})")

    # 10. Translation method check (Rule 7)
    if getattr(record, "translation_method", None) == "machine_only":
        r.append("machine_only_translation")

    return r

