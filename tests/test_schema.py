"""Tests for dataset schema validation and provenance integrity."""
from datetime import date

import pytest
from pydantic import ValidationError

from sierraleone_ai.datasets.schema import (
    AgriculturalConversationRecord,
    ConversationRecord,
    ConversationTurn,
    DatasetMetadata,
    KrioTextRecord,
    Language,
    LicenceType,
    ReviewerSignoff,
    TranslationRecord,
    ValidationStatus,
    VerifiedSpeechTranscriptionRecord,
)


def _make_reviewers(n: int = 2) -> list[ReviewerSignoff]:
    """Create n distinct native-speaker approvals for test fixtures."""
    return [
        ReviewerSignoff(
            reviewer_id=f"TEST_REVIEWER_{i:02d}",
            is_native_sl_krio_speaker=True,
            reviewed_on=date(2024, 1, 1),
            verdict="approve",
        )
        for i in range(1, n + 1)
    ]


def _make_metadata(**kwargs) -> DatasetMetadata:
    """
    Build a DatasetMetadata for tests.

    When validation_status=VALIDATED we include 2 reviewer sign-offs automatically
    unless the caller supplies reviews= explicitly.
    """
    status = kwargs.get("validation_status", ValidationStatus.UNVALIDATED)
    defaults: dict = dict(
        language=Language.KRIO,
        source="test_source",
        collection_date=date(2024, 1, 1),
        licence=LicenceType.CC_BY_4,
        validation_status=status,
    )
    if status == ValidationStatus.VALIDATED and "reviews" not in kwargs:
        defaults["reviews"] = _make_reviewers(2)
    if status == ValidationStatus.PARTIALLY_VALIDATED and "reviews" not in kwargs:
        defaults["reviews"] = _make_reviewers(1)
    defaults.update(kwargs)
    return DatasetMetadata(**defaults)


class TestDatasetMetadata:
    def test_valid_metadata(self):
        m = _make_metadata()
        assert m.language == Language.KRIO
        assert m.validation_status == ValidationStatus.UNVALIDATED

    def test_source_cannot_be_empty(self):
        with pytest.raises((ValidationError, ValueError)):
            _make_metadata(source="   ")

    def test_record_id_is_uuid(self):
        m = _make_metadata()
        assert len(m.record_id) == 36

    def test_validated_requires_two_native_reviewers(self):
        with pytest.raises(Exception, match="2 distinct"):
            _make_metadata(
                validation_status=ValidationStatus.VALIDATED,
                reviews=_make_reviewers(1),  # only 1 — should fail
            )

    def test_validated_with_two_reviewers_passes(self):
        m = _make_metadata(
            validation_status=ValidationStatus.VALIDATED,
            reviews=_make_reviewers(2),
        )
        assert m.validation_status == ValidationStatus.VALIDATED

    def test_consent_scope_required_when_consent_true(self):
        with pytest.raises(Exception, match="consent_scope"):
            _make_metadata(contributor_consent=True, consent_scope=[])

    def test_consent_scope_accepted_when_provided(self):
        m = _make_metadata(
            contributor_consent=True,
            consent_scope=["training", "evaluation"],
        )
        assert "training" in m.consent_scope


class TestKrioTextRecord:
    def test_valid_record(self):
        # FIXTURE — not real Krio; see docs/NATIVE_REVIEW_QUEUE.md
        r = KrioTextRecord(
            metadata=_make_metadata(),
            text="TEST_FIXTURE_TEXT_001",
            domain="general",
        )
        assert r.text == "TEST_FIXTURE_TEXT_001"

    def test_empty_text_rejected(self):
        with pytest.raises((ValidationError, ValueError)):
            KrioTextRecord(metadata=_make_metadata(), text="")


class TestTranslationRecord:
    def test_valid_translation(self):
        r = TranslationRecord(
            metadata=_make_metadata(),
            source_language=Language.ENGLISH,
            target_language=Language.KRIO,
            source_text="FIXTURE_EN_GOOD_MORNING",
            target_text="FIXTURE_KR_PLACEHOLDER_001",  # UNVERIFIED – see docs/NATIVE_REVIEW_QUEUE.md
            translation_method="human",
        )
        assert r.translation_method == "human"

    def test_invalid_translation_method(self):
        with pytest.raises((ValidationError, ValueError)):
            TranslationRecord(
                metadata=_make_metadata(),
                source_language=Language.ENGLISH,
                target_language=Language.KRIO,
                source_text="FIXTURE_EN_HELLO",
                target_text="FIXTURE_KR_PLACEHOLDER_002",
                translation_method="google_translate",
            )

    def test_machine_only_cannot_be_validated(self):
        with pytest.raises((ValidationError, ValueError)):
            TranslationRecord(
                metadata=_make_metadata(validation_status=ValidationStatus.VALIDATED),
                source_language=Language.ENGLISH,
                target_language=Language.KRIO,
                source_text="FIXTURE_EN_HELLO",
                target_text="FIXTURE_KR_PLACEHOLDER_002",
                translation_method="machine_only",
            )


class TestConversationAndAgriculturalRecords:
    def test_conversation_num_turns(self):
        # FIXTURE TEXT — not real Krio; see docs/NATIVE_REVIEW_QUEUE.md
        turns = [
            ConversationTurn(
                speaker="user", language=Language.KRIO,
                text="TEST_FIXTURE_TURN_A", turn_index=0,
            ),
            ConversationTurn(
                speaker="assistant", language=Language.KRIO,
                text="TEST_FIXTURE_TURN_B", turn_index=1,
            ),
        ]
        conv = ConversationRecord(
            metadata=_make_metadata(),
            domain="general",
            turns=turns,
        )
        assert conv.num_turns == 2

    def test_agricultural_conversation_record(self):
        # FIXTURE TEXT — not real Krio; see docs/NATIVE_REVIEW_QUEUE.md
        turns = [
            ConversationTurn(
                speaker="user", language=Language.KRIO,
                text="TEST_FIXTURE_AGRI_Q", turn_index=0,
            ),
            ConversationTurn(
                speaker="assistant", language=Language.KRIO,
                text="TEST_FIXTURE_AGRI_A", turn_index=1,
            ),
        ]
        agri = AgriculturalConversationRecord(
            metadata=_make_metadata(),
            crop_or_livestock="cassava",
            agro_ecological_zone="upland",
            season="rainy_season",
            agronomist_verified=True,
            turns=turns,
        )
        assert agri.domain == "agriculture"
        assert agri.agronomist_verified is True


class TestVerifiedSpeechTranscriptionRecord:
    def test_valid_verified_transcription(self):
        # FIXTURE — not real Krio; see docs/NATIVE_REVIEW_QUEUE.md
        rec = VerifiedSpeechTranscriptionRecord(
            metadata=_make_metadata(
                validation_status=ValidationStatus.VALIDATED,
                contributor_consent=True,
                consent_scope=["training"],
                reviews=_make_reviewers(2),
            ),
            audio_file_path="krio/audio/spk_01/rec_01.wav",
            transcript="TEST_FIXTURE_TRANSCRIPT_001",
            speaker_id="spk_01",
            transcriber_id="annotator_01",
            verifier_id="annotator_02",
        )
        assert rec.speaker_id == "spk_01"

    def test_same_transcriber_and_verifier_rejected(self):
        with pytest.raises((ValidationError, ValueError)):
            VerifiedSpeechTranscriptionRecord(
                metadata=_make_metadata(
                    validation_status=ValidationStatus.VALIDATED,
                    reviews=_make_reviewers(2),
                ),
                audio_file_path="krio/audio/spk_01/rec_01.wav",
                transcript="TEST_FIXTURE_TRANSCRIPT_002",
                speaker_id="spk_01",
                transcriber_id="annotator_01",
                verifier_id="annotator_01",
            )

    def test_partially_validated_rejected_for_speech(self):
        """VerifiedSpeechTranscriptionRecord must reject PARTIALLY_VALIDATED status."""
        with pytest.raises((ValidationError, ValueError)):
            VerifiedSpeechTranscriptionRecord(
                metadata=_make_metadata(
                    validation_status=ValidationStatus.PARTIALLY_VALIDATED,
                    reviews=_make_reviewers(1),
                ),
                audio_file_path="krio/audio/spk_01/rec_01.wav",
                transcript="TEST_FIXTURE_TRANSCRIPT_003",
                speaker_id="spk_01",
                transcriber_id="annotator_01",
                verifier_id="annotator_02",
            )
