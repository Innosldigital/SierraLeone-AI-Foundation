"""Tests for speaker leakage prevention, ASR/TTS infrastructure, and training guard."""
from datetime import date

import pytest

from sierraleone_ai.datasets.schema import (
    AudioRecord,
    ConsentScope,
    DatasetMetadata,
    Language,
    LicenceType,
    ReviewerSignoff,
    ValidationStatus,
    VerifiedSpeechTranscriptionRecord,
)
from sierraleone_ai.preprocessing.split_manager import speaker_aware_split
from sierraleone_ai.speech.asr import audit_speaker_leakage
from sierraleone_ai.speech.tts import check_tts_recording_quality
from sierraleone_ai.training.guard import require_training_approval


def _make_reviewers(n: int = 2) -> list[ReviewerSignoff]:
    return [
        ReviewerSignoff(
            reviewer_id=f"TEST_REVIEWER_{i:02d}",
            is_native_sl_krio_speaker=True,
            reviewed_on=date(2024, 1, 1),
            verdict="approve",
        )
        for i in range(1, n + 1)
    ]


_DEFAULT_CONSENT_SCOPE: list[ConsentScope] = ["training"]


def _make_validated_metadata(**kwargs) -> DatasetMetadata:
    defaults: dict = dict(
        language=Language.KRIO,
        source="test_audio",
        collection_date=date(2024, 1, 1),
        licence=LicenceType.CC_BY_4,
        validation_status=ValidationStatus.VALIDATED,
        contributor_consent=True,
        consent_scope=_DEFAULT_CONSENT_SCOPE,
        reviews=_make_reviewers(2),
    )
    defaults.update(kwargs)
    return DatasetMetadata(**defaults)


def _make_audio_record(speaker_id: str) -> AudioRecord:
    return AudioRecord(
        metadata=_make_validated_metadata(),
        audio_file_path=f"krio/audio/{speaker_id}/rec.wav",
        speaker_id=speaker_id,
    )


def test_speaker_aware_split_prevents_leakage():
    records = [{"speaker_id": f"spk_{i // 3}", "id": i} for i in range(30)]
    splits = speaker_aware_split(records, seed=42)
    train_spk = {r["speaker_id"] for r in splits["train"]}
    val_spk = {r["speaker_id"] for r in splits["validation"]}
    test_spk = {r["speaker_id"] for r in splits["test"]}
    assert train_spk.isdisjoint(val_spk)
    assert train_spk.isdisjoint(test_spk)
    assert val_spk.isdisjoint(test_spk)


def test_audit_speaker_leakage_detects_overlap():
    splits = {
        "train": [_make_audio_record("spk_01"), _make_audio_record("spk_02")],
        "validation": [_make_audio_record("spk_03")],
        "test": [_make_audio_record("spk_01")],  # intentional leak
    }
    report = audit_speaker_leakage(splits)
    assert report.has_leakage is True
    assert "spk_01" in report.leaked_train_test


def test_tts_recording_quality_check_passes():
    """A complete, studio-quality recording must pass the TTS quality gate."""
    rec = VerifiedSpeechTranscriptionRecord(
        metadata=_make_validated_metadata(),
        audio_file_path="krio/audio/spk_01/rec.wav",
        transcript="TEST_FIXTURE_TRANSCRIPT_001",  # UNVERIFIED – see docs/NATIVE_REVIEW_QUEUE.md
        speaker_id="spk_01",
        transcriber_id="t1",
        verifier_id="v1",
        sample_rate_hz=24000,
        snr_db=30.0,
        recording_environment="studio",
        duration_seconds=2.5,
    )
    eligible, issues = check_tts_recording_quality(rec)
    assert eligible is True
    assert issues == []


def test_tts_recording_quality_check_fails_missing_snr():
    """A recording with no SNR measurement must fail the TTS quality gate."""
    rec = VerifiedSpeechTranscriptionRecord(
        metadata=_make_validated_metadata(),
        audio_file_path="krio/audio/spk_02/rec.wav",
        transcript="TEST_FIXTURE_TRANSCRIPT_002",
        speaker_id="spk_02",
        transcriber_id="t1",
        verifier_id="v1",
        sample_rate_hz=24000,
        snr_db=None,  # missing
        recording_environment="studio",
        duration_seconds=2.0,
    )
    eligible, issues = check_tts_recording_quality(rec)
    assert eligible is False
    assert any("SNR not measured" in issue for issue in issues)


def test_tts_recording_quality_check_fails_missing_environment():
    """A recording with no environment metadata must fail the TTS quality gate."""
    rec = VerifiedSpeechTranscriptionRecord(
        metadata=_make_validated_metadata(),
        audio_file_path="krio/audio/spk_03/rec.wav",
        transcript="TEST_FIXTURE_TRANSCRIPT_003",
        speaker_id="spk_03",
        transcriber_id="t1",
        verifier_id="v1",
        sample_rate_hz=24000,
        snr_db=30.0,
        recording_environment=None,  # missing
        duration_seconds=2.0,
    )
    eligible, issues = check_tts_recording_quality(rec)
    assert eligible is False
    assert any("environment not recorded" in issue for issue in issues)


def test_tts_recording_quality_check_fails_missing_duration():
    """A recording with no duration measurement must fail the TTS quality gate."""
    rec = VerifiedSpeechTranscriptionRecord(
        metadata=_make_validated_metadata(),
        audio_file_path="krio/audio/spk_04/rec.wav",
        transcript="TEST_FIXTURE_TRANSCRIPT_004",
        speaker_id="spk_04",
        transcriber_id="t1",
        verifier_id="v1",
        sample_rate_hz=24000,
        snr_db=30.0,
        recording_environment="studio",
        duration_seconds=None,  # missing
    )
    eligible, issues = check_tts_recording_quality(rec)
    assert eligible is False
    assert any("duration not measured" in issue for issue in issues)


def test_training_guard_blocks_by_default():
    with pytest.raises(RuntimeError, match="TRAINING GUARD"):
        require_training_approval("unit test training")
