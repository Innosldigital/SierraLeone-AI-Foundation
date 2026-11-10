"""
Standardized dataset schemas for Sierra Leonean language data.

Every record in every dataset MUST include the metadata fields defined here.
This prevents mixing of verified and unverified data and ensures traceability.
"""
from __future__ import annotations

import uuid
from datetime import date
from enum import Enum
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
    UNKNOWN = "unknown"


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
# Shared metadata
# ---------------------------------------------------------------------------

class DatasetMetadata(BaseModel):
    """Mandatory provenance metadata attached to every dataset record."""

    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    language: Language
    source: str = Field(
        ...,
        description="Where the data came from, e.g. 'manual_collection_cafp_2024'",
    )
    collection_date: date
    licence: LicenceType
    validation_status: ValidationStatus = ValidationStatus.UNVALIDATED
    dataset_version: str = Field("0.1.0", pattern=r"^\d+\.\d+\.\d+$")
    split: DataSplit = DataSplit.TRAIN
    contributor_consent: bool = Field(
        False,
        description="True if contributor has given explicit informed consent",
    )
    # Structured consent metadata (required when contributor_consent=True)
    consent_scope: list[Literal["training", "evaluation", "publication", "commercial"]] = Field(
        default_factory=list,
        description="What the contributor has consented to. Required when contributor_consent=True.",
    )
    consent_record_id: str | None = Field(
        None,
        description="ID of the signed consent form in the consent management system",
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

    @field_validator("source")
    @classmethod
    def source_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("source must not be empty")
        return v.strip()

    @model_validator(mode="after")
    def enforce_review_and_consent_rules(self) -> DatasetMetadata:
        # Consent scope required when consent is given
        if self.contributor_consent and not self.consent_scope:
            raise ValueError(
                "consent_scope must be provided when contributor_consent=True. "
                "Specify what the contributor has consented to (e.g. ['training', 'evaluation'])."
            )
        # VALIDATED requires ≥2 distinct native-speaker approvals
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
    def prevent_machine_only_as_validated(self) -> TranslationRecord:
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

    speaker: str = Field(..., description="'user', 'assistant', or a speaker label")
    language: Language
    text: str = Field(..., min_length=1)
    turn_index: int = Field(..., ge=0)


class ConversationRecord(BaseModel):
    """A multi-turn dialogue."""

    metadata: DatasetMetadata
    conversation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    domain: str
    turns: list[ConversationTurn] = Field(..., min_length=2)
    num_turns: int = 0

    def model_post_init(self, __context) -> None:  # noqa: ANN001
        self.num_turns = len(self.turns)


class AgriculturalConversationRecord(ConversationRecord):
    """
    A multi-turn dialogue in the agricultural domain for CAFP Krio AI.
    Tracks crop/livestock focus, Sierra Leone agro-ecological context, and agronomic verification.
    """

    domain: str = Field("agriculture", frozen=True)
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
# Audio record schemas
# ---------------------------------------------------------------------------

class AudioRecord(BaseModel):
    """Metadata for a single audio recording."""

    metadata: DatasetMetadata
    audio_file_path: str = Field(
        ...,
        description="Relative path from datasets root, e.g. 'krio/audio/speaker_01/rec_001.wav'",
    )
    transcript: str | None = None
    duration_seconds: float | None = Field(None, ge=0)
    sample_rate_hz: int | None = Field(None, description="e.g. 16000, 44100")
    speaker_id: str | None = Field(
        None,
        description="Anonymised speaker identifier – used to prevent speaker leakage across splits",
    )
    gender: str | None = None
    age_group: str | None = None
    recording_environment: str | None = Field(
        None, description="'studio', 'field', 'phone', 'home'"
    )
    snr_db: float | None = Field(
        None, description="Signal-to-noise ratio in dB, if measured"
    )


class VerifiedSpeechTranscriptionRecord(AudioRecord):
    """
    A verified audio-transcription pair for ASR/TTS training and evaluation.
    Enforces speaker_id presence (to prevent split leakage) and dual human verification.
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
        if self.metadata.validation_status in (
            ValidationStatus.UNVALIDATED,
            ValidationStatus.MACHINE_GENERATED,
            ValidationStatus.REJECTED,
            ValidationStatus.PARTIALLY_VALIDATED,
        ):
            raise ValueError(
                f"VerifiedSpeechTranscriptionRecord cannot have status "
                f"'{self.metadata.validation_status.value}'. "
                "Only ValidationStatus.VALIDATED is accepted — this requires ≥2 "
                "independent native-speaker reviews recorded in metadata.reviews."
            )
        return self
