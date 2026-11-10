"""
Standardized dataset schemas for Sierra Leonean language data.

Every record in every dataset MUST include the metadata fields defined here.
This prevents mixing of verified and unverified data and ensures traceability.
"""
from __future__ import annotations

import hashlib
import re
import uuid
from datetime import date
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Language(str, Enum):
    KRIO = "krio"
    MENDE = "mende"
    TEMNE = "temne"
    LIMBA = "limba"
    LOKO = "loko"
    KONO = "kono"
    KISSI = "kissi"
    MANDINGO = "mandingo"
    ENGLISH = "english"
    OTHER = "other"


class ValidationStatus(str, Enum):
    UNVALIDATED = "unvalidated"          # raw, not yet reviewed
    MACHINE_GENERATED = "machine_generated"  # produced by MT or TTS – NOT ground truth
    PARTIALLY_VALIDATED = "partially_validated"  # reviewed by one native speaker
    VALIDATED = "validated"              # reviewed by ≥2 independent native speakers
    REJECTED = "rejected"               # found to contain errors; do not use for training


class DataSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    HELD_OUT = "held_out"               # permanent evaluation set – never trained on


class LicenceType(str, Enum):
    CC_BY_4 = "CC-BY-4.0"
    CC_BY_SA_4 = "CC-BY-SA-4.0"
    CC0 = "CC0-1.0"
    APACHE_2 = "Apache-2.0"
    MIT = "MIT"
    PROPRIETARY = "proprietary"
    CUSTOM = "custom"
    UNKNOWN = "unknown"


ALLOWED_LICENCE_IDS: frozenset[str] = frozenset({
    "CC-BY-4.0",
    "CC-BY-SA-4.0",
    "CC0-1.0",
    "Apache-2.0",
    "MIT",
    "proprietary",
    "custom",
    "unknown",
})

ConsentScope = Literal[
    "text_training",
    "speech_recognition",
    "voice_synthesis",
    "public_release",
    "commercial_use",
    # Backward-compatible aliases used in existing tests/configs
    "training",
    "evaluation",
    "publication",
    "commercial",
]

AccessClass = Literal["public", "restricted", "internal"]

_VALID_SPEAKER_ROLE_RE = re.compile(
    r"^(user|assistant|system|farmer|extension_officer|speaker_[a-zA-Z0-9_]+|participant_[a-zA-Z0-9_]+|annotator_[a-zA-Z0-9_]+)$"
)


# ---------------------------------------------------------------------------
# Reviewer sign-off (SLAI-013: enforce dual native-speaker review)
# ---------------------------------------------------------------------------

class ReviewerSignoff(BaseModel):
    """Record of a single native-speaker reviewer's verdict on a dataset record."""

    reviewer_id: str = Field(..., min_length=1, description="Anonymised reviewer identifier")
    is_native_sl_krio_speaker: bool = Field(
        ..., description="Must be True for VALIDATED status"
    )
    reviewed_on: date
    verdict: Literal["approve", "reject"]


# ---------------------------------------------------------------------------
# Shared metadata (SLAI-013 / SLAI-025 / SLAI-028)
# ---------------------------------------------------------------------------

class DatasetMetadata(BaseModel):
    """Mandatory provenance and consent metadata attached to every dataset record."""

    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    language: Language
    source: str = Field(
        ...,
        description="Where the data came from, e.g. 'manual_collection_cafp_2024'",
    )
    collection_date: date
    licence: LicenceType = LicenceType.UNKNOWN
    licence_id: str | None = Field(
        default=None,
        description="SPDX licence identifier or 'custom'/'proprietary'/'unknown' (SLAI-025)",
    )
    licence_url: str | None = Field(
        default=None,
        description="URL to the licence text or agreement",
    )
    rights_holder: str | None = Field(
        default=None,
        description="Copyright or data rights holder (SLAI-025)",
    )
    source_url_or_citation: str | None = Field(
        default=None,
        description="Canonical URL or bibliographic citation of the source work (SLAI-025)",
    )
    acquisition_date: date | None = Field(
        default=None,
        description="Date when the dataset or source material was acquired",
    )
    redistribution_permitted: bool = Field(
        default=True,
        description="True only if redistribution is legally permitted under the source licence",
    )
    training_permitted: bool = Field(
        default=True,
        description="True only if model training is permitted under the source licence and consent",
    )
    access_class: AccessClass = Field(
        default="internal",
        description="Access tier: 'public', 'restricted', or 'internal'",
    )
    restrictions: list[str] = Field(
        default_factory=list,
        description="Specific usage or redistribution restrictions on this record",
    )
    validation_status: ValidationStatus = ValidationStatus.UNVALIDATED
    dataset_version: str = Field("0.1.0", pattern=r"^\d+\.\d+\.\d+$")
    split: DataSplit = DataSplit.TRAIN
    contributor_consent: bool = Field(
        False,
        description="True if contributor has given explicit informed consent",
    )
    # Structured consent metadata (required when contributor_consent=True)
    consent_scope: list[ConsentScope] = Field(
        default_factory=list,
        description="What the contributor has consented to. Required when contributor_consent=True.",
    )
    consent_record_id: str | None = Field(
        None,
        description="ID of the signed consent form in the restricted consent store (never store PII in JSONL)",
    )
    consent_withdrawn: bool = Field(
        False,
        description="Set True if contributor has subsequently withdrawn consent. Record must be removed from training.",
    )
    # Reviewer sign-offs (required for VALIDATED and PARTIALLY_VALIDATED status)
    reviews: list[ReviewerSignoff] = Field(
        default_factory=list,
        description="List of native-speaker reviewer sign-offs. VALIDATED requires ≥2 distinct native approvals.",
    )
    notes: str | None = None

    @field_validator("record_id")
    @classmethod
    def validate_uuid_format(cls, v: str) -> str:
        try:
            uuid.UUID(v)
        except ValueError as exc:
            raise ValueError(f"record_id must be a valid UUID string, got '{v}'") from exc
        return v

    @field_validator("source")
    @classmethod
    def source_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("source must not be empty")
        return v.strip()

    @field_validator("collection_date")
    @classmethod
    def collection_date_not_in_future(cls, v: date) -> date:
        if v > date.today():
            raise ValueError(f"collection_date ({v}) cannot be in the future")
        return v

    @field_validator("licence_id")
    @classmethod
    def validate_licence_id(cls, v: str | None) -> str | None:
        if v is not None and v not in ALLOWED_LICENCE_IDS:
            raise ValueError(
                f"licence_id must be one of {sorted(ALLOWED_LICENCE_IDS)}, got '{v}'"
            )
        return v

    @model_validator(mode="after")
    def enforce_review_and_consent_rules(self) -> DatasetMetadata:
        # Synchronise licence_id and licence
        if self.licence_id is None:
            self.licence_id = self.licence.value
        elif self.licence == LicenceType.UNKNOWN and self.licence_id != "unknown":
            for member in LicenceType:
                if member.value == self.licence_id:
                    self.licence = member
                    break

        # Unknown rights must never be marked as permitted for training or redistribution (SLAI-025)
        if self.licence == LicenceType.UNKNOWN or self.licence_id == "unknown":
            self.training_permitted = False
            self.redistribution_permitted = False

        # Consent scope required when consent is given
        if self.contributor_consent and not self.consent_scope:
            raise ValueError(
                "consent_scope must be provided when contributor_consent=True. "
                "Specify what the contributor has consented to (e.g. ['text_training', 'speech_recognition'])."
            )

        # VALIDATED requires ≥2 distinct native-speaker approvals, consent=True, and known licence
        if self.validation_status == ValidationStatus.VALIDATED:
            native_approvals = [
                r for r in self.reviews
                if r.is_native_sl_krio_speaker and r.verdict == "approve"
            ]
            reviewer_ids = {r.reviewer_id for r in native_approvals}
            if len(reviewer_ids) < 2:
                raise ValueError(
                    "ValidationStatus.VALIDATED requires reviews from ≥2 distinct native "
                    "Sierra Leonean Krio speakers with verdict='approve'. "
                    f"Found {len(reviewer_ids)} qualifying reviewer(s)."
                )
            if not self.contributor_consent:
                raise ValueError(
                    "ValidationStatus.VALIDATED requires contributor_consent=True."
                )
            if self.licence == LicenceType.UNKNOWN or self.licence_id == "unknown":
                raise ValueError(
                    "ValidationStatus.VALIDATED cannot have an 'unknown' licence."
                )
        # PARTIALLY_VALIDATED requires ≥1 native-speaker approval
        elif self.validation_status == ValidationStatus.PARTIALLY_VALIDATED:
            native_approvals = [
                r for r in self.reviews
                if r.is_native_sl_krio_speaker and r.verdict == "approve"
            ]
            if not native_approvals:
                raise ValueError(
                    "ValidationStatus.PARTIALLY_VALIDATED requires at least 1 native "
                    "Sierra Leonean Krio speaker review with verdict='approve'."
                )
        return self


# ---------------------------------------------------------------------------
# Text record schemas
# ---------------------------------------------------------------------------

class KrioTextRecord(BaseModel):
    """A single Krio text sample."""

    metadata: DatasetMetadata
    text: str = Field(..., min_length=1)
    script: str = Field("Latin", description="Writing system used")
    domain: str | None = Field(
        None,
        description="Thematic domain, e.g. 'agriculture', 'health', 'general'",
    )


class TranslationRecord(BaseModel):
    """A bilingual translation pair."""

    metadata: DatasetMetadata
    source_language: Language
    target_language: Language
    source_text: str = Field(..., min_length=1)
    target_text: str = Field(..., min_length=1)
    translation_method: str = Field(
        ...,
        description="'human', 'machine_then_human_post_edit', or 'machine_only'",
    )
    back_translation_verified: bool = False

    @field_validator("translation_method")
    @classmethod
    def valid_method(cls, v: str) -> str:
        allowed = {"human", "machine_then_human_post_edit", "machine_only"}
        if v not in allowed:
            raise ValueError(f"translation_method must be one of {allowed}")
        return v

    @model_validator(mode="after")
    def validate_translation_integrity(self) -> TranslationRecord:
        if self.source_language == self.target_language:
            raise ValueError(
                f"source_language and target_language must differ (got '{self.source_language.value}')"
            )
        if self.metadata.language not in (self.source_language, self.target_language):
            raise ValueError(
                f"metadata.language ('{self.metadata.language.value}') must match either "
                f"source_language ('{self.source_language.value}') or "
                f"target_language ('{self.target_language.value}')"
            )
        if (
            self.translation_method == "machine_only"
            and self.metadata.validation_status == ValidationStatus.VALIDATED
        ):
            raise ValueError(
                "machine_only translations cannot have validation_status='validated'. "
                "Never represent machine-generated translations as native-speaker-verified data."
            )
        return self


class ConversationTurn(BaseModel):
    """A single turn in a conversation."""

    speaker: str = Field(..., min_length=1, description="'user', 'assistant', or a speaker label")
    language: Language
    text: str = Field(..., min_length=1)
    turn_index: int = Field(..., ge=0)

    @field_validator("speaker")
    @classmethod
    def validate_speaker_role(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned or not _VALID_SPEAKER_ROLE_RE.match(cleaned):
            raise ValueError(
                f"Invalid speaker role '{v}'. Allowed roles: 'user', 'assistant', 'system', "
                "'farmer', 'extension_officer', or 'speaker_<id>' / 'participant_<id>'."
            )
        return cleaned


class ConversationRecord(BaseModel):
    """A multi-turn dialogue."""

    metadata: DatasetMetadata
    conversation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    domain: str
    turns: list[ConversationTurn] = Field(..., min_length=2)
    num_turns: int = 0

    @field_validator("conversation_id")
    @classmethod
    def validate_conversation_uuid(cls, v: str) -> str:
        try:
            uuid.UUID(v)
        except ValueError as exc:
            raise ValueError(f"conversation_id must be a valid UUID string, got '{v}'") from exc
        return v

    @model_validator(mode="after")
    def validate_turns_integrity(self) -> ConversationRecord:
        indices = [t.turn_index for t in self.turns]
        expected = list(range(len(self.turns)))
        if indices != expected:
            raise ValueError(
                f"Conversation turn_index values must be contiguous and ordered starting from 0 "
                f"(expected {expected}, got {indices})"
            )
        if self.metadata.language != Language.OTHER:
            if not any(t.language == self.metadata.language for t in self.turns):
                raise ValueError(
                    f"No turn in ConversationRecord matches metadata.language "
                    f"('{self.metadata.language.value}')"
                )
        self.num_turns = len(self.turns)
        return self


class AgriculturalConversationRecord(ConversationRecord):
    """
    A multi-turn dialogue in the agricultural domain for CAFP Krio AI.
    Tracks crop/livestock focus, Sierra Leone agro-ecological context, and agronomic verification.
    """

    domain: str = Field(default="agriculture")

    @field_validator("domain")
    @classmethod
    def enforce_agriculture_domain(cls, v: str) -> str:
        if v != "agriculture":
            raise ValueError("AgriculturalConversationRecord domain must be 'agriculture'")
        return v

    crop_or_livestock: str | None = Field(
        None,
        description="Target crop or livestock, e.g. 'cassava (kasada)', 'rice (rais)', 'cocoa', 'poultry'",
    )
    agro_ecological_zone: str | None = Field(
        None,
        description="Sierra Leone zone, e.g. 'boliland', 'inland_valley_swamp', 'upland', 'mangrove', 'coastal'",
    )
    season: str | None = Field(
        None,
        description="'rainy_season (rein sizon)', 'dry_season (drai sizon)', or 'harmattan'",
    )
    agronomist_verified: bool = Field(
        False,
        description="True only if an agricultural extension officer or agronomist verified the advice",
    )


class QAPairRecord(BaseModel):
    """A question-answer pair."""

    metadata: DatasetMetadata
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    question_language: Language
    answer_language: Language
    domain: str | None = None
    is_factual: bool = True


# ---------------------------------------------------------------------------
# Audio record schemas (SLAI-026 / SLAI-028)
# ---------------------------------------------------------------------------

class AudioRecord(BaseModel):
    """Metadata for a single audio recording."""

    metadata: DatasetMetadata
    audio_file_path: str = Field(
        ...,
        min_length=1,
        description="Relative path from datasets root, e.g. 'krio/audio/speaker_01/rec_001.wav'",
    )
    audio_sha256: str | None = Field(
        default=None,
        pattern=r"^[a-fA-F0-9]{64}$",
        description="Optional SHA-256 hex digest of the audio file (SLAI-028)",
    )
    transcript: str | None = None
    duration_seconds: float | None = Field(default=None, ge=0)
    sample_rate_hz: int | None = Field(default=None, description="e.g. 16000, 44100")
    speaker_id: str | None = Field(
        default=None,
        description="Anonymised speaker identifier – used to prevent speaker leakage across splits",
    )
    gender: str | None = None
    age_group: str | None = None
    recording_environment: str | None = Field(
        default=None, description="'studio', 'field', 'phone', 'home'"
    )
    snr_db: float | None = Field(
        default=None, description="Signal-to-noise ratio in dB, if measured"
    )

    @field_validator("audio_file_path")
    @classmethod
    def validate_safe_relative_audio_path(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("audio_file_path must not be empty")
        if (
            cleaned.startswith(("/", "\\"))
            or ":" in cleaned
            or PurePosixPath(cleaned).is_absolute()
            or PureWindowsPath(cleaned).is_absolute()
        ):
            raise ValueError(
                f"audio_file_path must be a relative path under datasets root, got '{v}'"
            )
        parts = PurePosixPath(cleaned.replace("\\", "/")).parts
        if ".." in parts:
            raise ValueError(
                f"Path traversal ('..') is forbidden in audio_file_path: '{v}'"
            )
        return cleaned

    def resolve_audio_path(self, dataset_root: Path | None = None) -> Path:
        """Resolve audio_file_path safely under dataset_root (SLAI-028)."""
        if dataset_root is None:
            from sierraleone_ai.config import settings
            dataset_root = settings.data_dir
        root_resolved = dataset_root.resolve()
        target = (root_resolved / self.audio_file_path).resolve()
        if not target.is_relative_to(root_resolved):
            raise ValueError(
                f"Resolved audio path '{target}' escapes dataset root '{root_resolved}'"
            )
        return target

    def verify_audio_file(self, dataset_root: Path | None = None) -> tuple[bool, str]:
        """Check that the audio file exists under dataset_root and matches audio_sha256 if set."""
        try:
            target = self.resolve_audio_path(dataset_root)
        except ValueError as exc:
            return False, str(exc)
        if not target.exists() or not target.is_file():
            return False, f"audio_file_missing({self.audio_file_path})"
        if self.audio_sha256 is not None:
            digest = hashlib.sha256(target.read_bytes()).hexdigest().lower()
            if digest != self.audio_sha256.lower():
                return False, "audio_sha256_mismatch"
        return True, ""


class VerifiedSpeechTranscriptionRecord(AudioRecord):
    """
    A verified audio-transcription pair for ASR/TTS training and evaluation.
    Enforces speaker_id presence (to prevent split leakage), dual human verification,
    ValidationStatus.VALIDATED, and explicit ASR/training consent scope (SLAI-026).
    """

    transcript: str = Field(..., min_length=1)
    speaker_id: str = Field(
        ...,
        min_length=1,
        description="Mandatory anonymised speaker ID to prevent train/test speaker leakage",
    )
    transcriber_id: str = Field(
        ..., min_length=1, description="Anonymised ID of the initial human transcriber"
    )
    verifier_id: str = Field(
        ..., min_length=1, description="Anonymised ID of the independent native-speaker verifier"
    )

    @model_validator(mode="after")
    def verify_transcription_integrity(self) -> VerifiedSpeechTranscriptionRecord:
        if self.transcriber_id == self.verifier_id:
            raise ValueError(
                "transcriber_id and verifier_id must be two independent reviewers"
            )
        if self.metadata.validation_status != ValidationStatus.VALIDATED:
            raise ValueError(
                f"VerifiedSpeechTranscriptionRecord cannot have status "
                f"'{self.metadata.validation_status.value}'. "
                "Only ValidationStatus.VALIDATED is accepted — this requires ≥2 "
                "independent native-speaker reviews recorded in metadata.reviews."
            )
        asr_scopes = {"speech_recognition", "training"}
        if not any(scope in asr_scopes for scope in self.metadata.consent_scope):
            raise ValueError(
                "VerifiedSpeechTranscriptionRecord requires 'speech_recognition' "
                "(or 'training') in metadata.consent_scope."
            )
        return self
