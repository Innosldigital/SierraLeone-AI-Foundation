"""Tests for dataset schema validation, consent/provenance model, and record integrity (SLAI-013 / SLAI-025 / SLAI-026 / SLAI-028)."""
from datetime import date, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from sierraleone_ai.datasets.schema import (
    AgriculturalConversationRecord,
    AudioRecord,
    ConsentScope,
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

_DEFAULT_CONSENT_SCOPE: list[ConsentScope] = ["text_training", "speech_recognition", "training"]


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

    When validation_status=VALIDATED we include 2 reviewer sign-offs and
    contributor_consent=True automatically unless the caller supplies them explicitly.
    """
    status = kwargs.get("validation_status", ValidationStatus.UNVALIDATED)
    defaults: dict = dict(
        language=Language.KRIO,
        source="test_source",
        collection_date=date(2024, 1, 1),
        licence=LicenceType.CC_BY_4,
        validation_status=status,
    )
    if status == ValidationStatus.VALIDATED:
        if "reviews" not in kwargs:
            defaults["reviews"] = _make_reviewers(2)
        if "contributor_consent" not in kwargs:
            defaults["contributor_consent"] = True
        if "consent_scope" not in kwargs and defaults.get("contributor_consent", True):
            defaults["consent_scope"] = list(_DEFAULT_CONSENT_SCOPE)
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

    def test_invalid_uuid_record_id_rejected(self):
        with pytest.raises((ValidationError, ValueError), match="UUID"):
            _make_metadata(record_id="not-a-valid-uuid")

    def test_future_collection_date_rejected(self):
        future = date.today() + timedelta(days=5)
        with pytest.raises((ValidationError, ValueError), match="future"):
            _make_metadata(collection_date=future)

    def test_validated_requires_two_native_reviewers(self):
        with pytest.raises(Exception, match="2 distinct"):
            _make_metadata(
                validation_status=ValidationStatus.VALIDATED,
                reviews=_make_reviewers(1),  # only 1 — should fail
            )

    def test_validated_requires_contributor_consent(self):
        with pytest.raises((ValidationError, ValueError), match="contributor_consent"):
            _make_metadata(
                validation_status=ValidationStatus.VALIDATED,
                contributor_consent=False,
                consent_scope=[],
            )

    def test_validated_rejects_unknown_licence(self):
        with pytest.raises((ValidationError, ValueError), match="unknown"):
            _make_metadata(
                validation_status=ValidationStatus.VALIDATED,
                licence=LicenceType.UNKNOWN,
            )

    def test_unknown_licence_disables_training_and_redistribution(self):
        m = _make_metadata(licence=LicenceType.UNKNOWN)
        assert m.training_permitted is False
        assert m.redistribution_permitted is False

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
        scope: list[ConsentScope] = ["text_training", "public_release"]
        m = _make_metadata(
            contributor_consent=True,
            consent_scope=scope,
        )
        assert "text_training" in m.consent_scope


class TestKrioTextRecord:
    def test_valid_record(self):
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
            target_text="FIXTURE_KR_PLACEHOLDER_001",
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

    def test_same_source_and_target_language_rejected(self):
        with pytest.raises((ValidationError, ValueError), match="must differ"):
            TranslationRecord(
                metadata=_make_metadata(),
                source_language=Language.KRIO,
                target_language=Language.KRIO,
                source_text="FIXTURE_A",
                target_text="FIXTURE_B",
                translation_method="human",
            )

    def test_metadata_language_conflict_rejected(self):
        with pytest.raises((ValidationError, ValueError), match="metadata.language"):
            TranslationRecord(
                metadata=_make_metadata(language=Language.TEMNE),
                source_language=Language.ENGLISH,
                target_language=Language.KRIO,
                source_text="FIXTURE_A",
                target_text="FIXTURE_B",
                translation_method="human",
            )


class TestConversationAndAgriculturalRecords:
    def test_conversation_num_turns(self):
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

    def test_conversation_non_contiguous_turns_rejected(self):
        turns = [
            ConversationTurn(
                speaker="user", language=Language.KRIO,
                text="TEST_FIXTURE_TURN_A", turn_index=0,
            ),
            ConversationTurn(
                speaker="assistant", language=Language.KRIO,
                text="TEST_FIXTURE_TURN_B", turn_index=2,  # gap!
            ),
        ]
        with pytest.raises((ValidationError, ValueError), match="contiguous"):
            ConversationRecord(
                metadata=_make_metadata(),
                domain="general",
                turns=turns,
            )

    def test_conversation_language_mismatch_rejected(self):
        turns = [
            ConversationTurn(
                speaker="user", language=Language.MENDE,
                text="TEST_FIXTURE_TURN_A", turn_index=0,
            ),
            ConversationTurn(
                speaker="assistant", language=Language.MENDE,
                text="TEST_FIXTURE_TURN_B", turn_index=1,
            ),
        ]
        with pytest.raises((ValidationError, ValueError), match="metadata.language"):
            ConversationRecord(
                metadata=_make_metadata(language=Language.KRIO),
                domain="general",
                turns=turns,
            )

    def test_agricultural_conversation_record(self):
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
        rec = VerifiedSpeechTranscriptionRecord(
            metadata=_make_metadata(
                validation_status=ValidationStatus.VALIDATED,
                contributor_consent=True,
                consent_scope=_DEFAULT_CONSENT_SCOPE,
                reviews=_make_reviewers(2),
            ),
            audio_file_path="krio/audio/spk_01/rec_01.wav",
            transcript="TEST_FIXTURE_TRANSCRIPT_001",
            speaker_id="spk_01",
            transcriber_id="annotator_01",
            verifier_id="annotator_02",
        )
        assert rec.speaker_id == "spk_01"

    def test_audio_path_traversal_rejected(self):
        with pytest.raises((ValidationError, ValueError), match="traversal"):
            AudioRecord(
                metadata=_make_metadata(),
                audio_file_path="../secrets/audio.wav",
                speaker_id="spk_01",
            )

    def test_audio_absolute_path_rejected(self):
        with pytest.raises((ValidationError, ValueError), match="relative path"):
            AudioRecord(
                metadata=_make_metadata(),
                audio_file_path="/etc/passwd",
                speaker_id="spk_01",
            )

    def test_audio_verify_file_and_sha256(self, tmp_path: Path):
        audio_file = tmp_path / "krio" / "audio" / "spk_01" / "rec.wav"
        audio_file.parent.mkdir(parents=True)
        audio_file.write_bytes(b"RIFF....WAVEfmt ")
        import hashlib
        sha = hashlib.sha256(b"RIFF....WAVEfmt ").hexdigest()

        rec = AudioRecord(
            metadata=_make_metadata(),
            audio_file_path="krio/audio/spk_01/rec.wav",
            audio_sha256=sha,
            speaker_id="spk_01",
        )
        ok, err = rec.verify_audio_file(tmp_path)
        assert ok is True
        assert err == ""

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

    def test_speech_requires_asr_consent_scope(self):
        """VerifiedSpeechTranscriptionRecord must reject text_training-only consent scope."""
        with pytest.raises((ValidationError, ValueError), match="speech_recognition"):
            VerifiedSpeechTranscriptionRecord(
                metadata=_make_metadata(
                    validation_status=ValidationStatus.VALIDATED,
                    contributor_consent=True,
                    consent_scope=["text_training"],
                    reviews=_make_reviewers(2),
                ),
                audio_file_path="krio/audio/spk_01/rec_01.wav",
                transcript="TEST_FIXTURE_TRANSCRIPT_004",
                speaker_id="spk_01",
                transcriber_id="annotator_01",
                verifier_id="annotator_02",
            )
