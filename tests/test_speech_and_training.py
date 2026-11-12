"""Tests for speaker leakage prevention, ASR/TTS infrastructure, and training guard."""
import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

from sierraleone_ai.datasets.schema import (
    AudioRecord,
    ConsentScope,
    DatasetMetadata,
    DataSplit,
    Language,
    LicenceType,
    ReviewerSignoff,
    ValidationStatus,
    VerifiedSpeechTranscriptionRecord,
)
from sierraleone_ai.preprocessing.split_manager import speaker_aware_split
from sierraleone_ai.speech.asr import audit_speaker_leakage
from sierraleone_ai.speech.tts import check_tts_recording_quality
from sierraleone_ai.training.guard import check_dataset_readiness, require_training_approval


def _make_reviewers(n: int = 2) -> list[ReviewerSignoff]:
    return [
        ReviewerSignoff(
            reviewer_id=f"TEST_REVIEWER_{i:02d}",
            is_native_speaker_of_record_language=True,
            reviewed_on=date(2024, 1, 1),
            verdict="approve",
        )
        for i in range(1, n + 1)
    ]


_DEFAULT_CONSENT_SCOPE: list[ConsentScope] = ["training", "speech_recognition", "voice_synthesis"]


def _make_validated_metadata(**kwargs) -> DatasetMetadata:
    defaults: dict = dict(
        language=Language.KRIO,
        source="test_audio",
        collection_date=date(2024, 1, 1),
        licence=LicenceType.CC_BY_4,
        validation_status=ValidationStatus.VALIDATED,
        split=DataSplit.TRAIN,
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


def test_tts_rejects_asr_only_consent_without_voice_synthesis():
    """SLAI-025 / SLAI-013: ASR consent alone must NOT satisfy TTS voice_synthesis consent."""
    asr_only_scope: list[ConsentScope] = ["speech_recognition"]
    rec = VerifiedSpeechTranscriptionRecord(
        metadata=_make_validated_metadata(consent_scope=asr_only_scope),
        audio_file_path="krio/audio/spk_05/rec.wav",
        transcript="TEST_FIXTURE_TRANSCRIPT_005",
        speaker_id="spk_05",
        transcriber_id="t1",
        verifier_id="v1",
        sample_rate_hz=24000,
        snr_db=30.0,
        recording_environment="studio",
        duration_seconds=2.5,
    )
    eligible, issues = check_tts_recording_quality(rec)
    assert eligible is False
    assert any("voice_synthesis" in issue for issue in issues)


def test_training_guard_dry_run_makes_no_writes_or_downloads():
    """SLAI-024: dry_run=True returns readiness/manifest inspection without throwing or writing."""
    result = require_training_approval("dry_run_test", dry_run=True)
    assert result["dry_run"] is True
    assert result["ready_to_train"] is False
    assert isinstance(result["readiness"], dict)


def test_readiness_with_only_templates_is_not_ready():
    """With only template files in repo, all categories including asr_audio/tts_audio are NOT READY."""
    readiness = check_dataset_readiness("krio")
    assert all(item.ok is False for item in readiness.values())
    assert set(readiness.keys()) == {
        "cleaned_text",
        "translations",
        "conversations",
        "held_out_benchmark",
        "asr_audio",
        "tts_audio",
    }


def test_training_guard_requires_manifest_even_when_allow_training_and_data_ready(
    tmp_path: Path, monkeypatch
):
    """SLAI-024: SLAI_ALLOW_TRAINING=true + ready data STILL fails without base model + manifest."""
    import sierraleone_ai.config as cfg_module
    import sierraleone_ai.training.guard as guard_module
    from sierraleone_ai.training.guard import ReadinessItem

    monkeypatch.setattr(cfg_module.settings, "allow_training", True)
    monkeypatch.setattr(
        guard_module,
        "check_dataset_readiness",
        lambda lang="krio": {
            "cleaned_text": ReadinessItem("cleaned_text", 60000, 50000),
            "translations": ReadinessItem("translations", 15000, 10000),
            "conversations": ReadinessItem("conversations", 2000, 1000),
            "held_out_benchmark": ReadinessItem("held_out_benchmark", 2000, 1000),
            "asr_audio": ReadinessItem("asr_audio", 2000, 1000),
            "tts_audio": ReadinessItem("tts_audio", 1000, 500),
        },
    )
    # Without manifest_path -> raises RuntimeError
    with pytest.raises(RuntimeError, match="missing approval manifest"):
        require_training_approval("test_op")


def test_training_guard_rejects_manifest_referencing_raw_or_mismatched_sha256(
    tmp_path: Path, monkeypatch
):
    """SLAI-024: Manifest referencing raw/ or wrong SHA-256 must fail closed."""
    from sierraleone_ai.training.guard import verify_training_manifest

    clean_file = tmp_path / "datasets" / "krio" / "cleaned" / "train.jsonl"
    clean_file.parent.mkdir(parents=True)
    clean_file.write_text('{"x": 1}\n', encoding="utf-8")
    real_sha = hashlib.sha256(clean_file.read_bytes()).hexdigest()

    licence_review = {
        "reviewer": "project_owner",
        "reviewed_on": "2025-01-15",
        "licence": "Apache-2.0",
        "commercial_permitted": True,
        "intended_use": "research",
    }

    # 1. Manifest referencing raw/ directory must fail validation
    bad_raw_manifest = tmp_path / "manifest_raw.json"
    bad_raw_manifest.write_text(
        json.dumps(
            {
                "run_id": "run_raw",
                "task": "sft",
                "base_model_id": "Qwen/Qwen3-4B",
                "base_model_revision": "abc1234",
                "model_licence_review": licence_review,
                "dataset_versions": {"train": "0.1.0"},
                "dataset_sha256_hashes": {"datasets/krio/raw/uncleaned.jsonl": real_sha},
                "dataset_manifest_id": "krio-train-v1",
                "split_manifest_id": "speaker_splits_v1",
                "benchmark_id": "bench_v1",
                "estimated_gpu_hours": 1.0,
                "estimated_cost_usd": 2.0,
                "approver": "project_owner",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="raw"):
        verify_training_manifest(bad_raw_manifest, base_dir=tmp_path)

    # 2. Manifest with wrong SHA-256 must fail
    bad_sha_manifest = tmp_path / "manifest_bad_sha.json"
    bad_sha_manifest.write_text(
        json.dumps(
            {
                "run_id": "run_sha",
                "task": "sft",
                "base_model_id": "Qwen/Qwen3-4B",
                "base_model_revision": "abc1234",
                "model_licence_review": licence_review,
                "dataset_versions": {"train": "0.1.0"},
                "dataset_sha256_hashes": {
                    "datasets/krio/cleaned/train.jsonl": "0" * 64
                },
                "dataset_manifest_id": "krio-train-v1",
                "split_manifest_id": "speaker_splits_v1",
                "benchmark_id": "bench_v1",
                "estimated_gpu_hours": 1.0,
                "estimated_cost_usd": 2.0,
                "approver": "project_owner",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        verify_training_manifest(bad_sha_manifest, base_dir=tmp_path)


def test_model_licence_review_requires_commercial_permitted():
    """PA-010: ModelLicenceReview requires commercial_permitted=True when intended_use='commercial'."""
    from datetime import date

    from sierraleone_ai.training.guard import ModelLicenceReview

    r = ModelLicenceReview(
        reviewer="rev",
        reviewed_on=date(2026, 11, 12),
        licence="CC-BY-NC-4.0",
        commercial_permitted=False,
        intended_use="research",
    )
    assert r.intended_use == "research"

    with pytest.raises(ValueError, match="commercial_permitted must be True"):
        ModelLicenceReview(
            reviewer="rev",
            reviewed_on=date(2026, 11, 12),
            licence="CC-BY-NC-4.0",
            commercial_permitted=False,
            intended_use="commercial",
        )


def test_training_guard_dry_run_extended_plan():
    """PA-010: dry-run plan contains eligible_counts, gpu_available, downloads=0, writes=0."""
    from sierraleone_ai.training.guard import require_training_approval

    plan = require_training_approval("dry_run_task", dry_run=True)
    assert plan["downloads"] == 0
    assert plan["writes"] == 0
    assert isinstance(plan["eligible_counts"], dict)
    assert "text_training" in plan["eligible_counts"]
    assert "gpu_available" in plan
    assert "resolved_config_paths" in plan


def test_training_guard_rejects_manifest_for_different_model(
    tmp_path: Path, monkeypatch
):
    """PA-010: Approval manifest base_model_id != selected base model must raise RuntimeError."""
    import sierraleone_ai.config as cfg_module
    from sierraleone_ai.training.guard import require_training_approval

    monkeypatch.setattr(cfg_module.settings, "allow_training", True)
    monkeypatch.setattr(cfg_module.settings, "data_dir", tmp_path)
    monkeypatch.setattr(cfg_module.settings, "models_dir", tmp_path / "models")

    # Set up selected_base_model.json with a specific model
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True)
    (models_dir / "selected_base_model.json").write_text(
        json.dumps({
            "hf_repo": "Qwen/Qwen2.5-3B",
            "tokenizer_audit_completed": True,
            "native_benchmark_completed": True,
        }),
        encoding="utf-8",
    )

    # Set up valid dataset files and manifests
    krio_dir = tmp_path / "krio"
    clean_file = krio_dir / "validated" / "krio_text_train.jsonl"
    clean_file.parent.mkdir(parents=True)
    clean_file.write_text('{"text": "val"}\n', encoding="utf-8")
    real_sha = hashlib.sha256(clean_file.read_bytes()).hexdigest()

    # Split manifest
    splits_dir = krio_dir / "splits"
    splits_dir.mkdir(parents=True)
    (splits_dir / "speaker_splits_v1.json").write_text(
        json.dumps({"speaker_to_split": {}}), encoding="utf-8"
    )

    # Dataset manifest
    manifests_dir = tmp_path / "datasets" / "manifests"
    manifests_dir.mkdir(parents=True)
    ds_manifest = manifests_dir / "krio-train-v1.manifest.json"
    rel_clean_path = str(clean_file.relative_to(tmp_path)).replace("\\", "/")
    ds_manifest.write_text(
        json.dumps({
            "schema_version": 2,
            "language": "krio",
            "split": "train",
            "version": 1,
            "files": {rel_clean_path: real_sha},
            "eligible_counts": {"text_training": 100000},
            "skipped": {},
            "built_at": "2026-11-12T00:00:00Z",
        }),
        encoding="utf-8",
    )

    licence_review = {
        "reviewer": "reviewer1",
        "reviewed_on": "2026-11-12",
        "licence": "Apache-2.0",
        "commercial_permitted": True,
        "intended_use": "research",
    }
    approval_path = tmp_path / "approval.json"
    approval_path.write_text(
        json.dumps({
            "run_id": "run_diff_model",
            "task": "chat_llm_sft",
            "base_model_id": "Different/Model-7B",
            "base_model_revision": "abc1234",
            "model_licence_review": licence_review,
            "dataset_versions": {"train": "0.1.0"},
            "dataset_sha256_hashes": {rel_clean_path: real_sha},
            "dataset_manifest_id": "krio-train-v1",
            "split_manifest_id": "speaker_splits_v1",
            "benchmark_id": "bench_v1",
            "estimated_gpu_hours": 2.0,
            "estimated_cost_usd": 4.0,
            "approver": "project_owner",
        }),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="does not match selected base model"):
        require_training_approval(
            "test_task",
            language="krio",
            manifest_path=approval_path,
            task_key="chat_llm_sft",
            base_dir=tmp_path,
            check_readiness=False,
        )


def test_training_guard_rejects_file_not_covered_by_dataset_manifest(
    tmp_path: Path,
):
    """PA-010: Manifest referencing a file not covered by dataset manifest must raise."""
    from sierraleone_ai.training.guard import verify_training_manifest

    krio_dir = tmp_path / "krio"
    clean_file1 = krio_dir / "validated" / "covered.jsonl"
    clean_file1.parent.mkdir(parents=True)
    clean_file1.write_text('{"x": 1}\n', encoding="utf-8")
    sha1 = hashlib.sha256(clean_file1.read_bytes()).hexdigest()

    clean_file2 = krio_dir / "validated" / "uncovered.jsonl"
    clean_file2.write_text('{"x": 2}\n', encoding="utf-8")
    sha2 = hashlib.sha256(clean_file2.read_bytes()).hexdigest()

    rel_p1 = str(clean_file1.relative_to(tmp_path)).replace("\\", "/")
    rel_p2 = str(clean_file2.relative_to(tmp_path)).replace("\\", "/")

    # Split manifest
    splits_dir = krio_dir / "splits"
    splits_dir.mkdir(parents=True)
    (splits_dir / "speaker_splits_v1.json").write_text(
        json.dumps({"speaker_to_split": {}}), encoding="utf-8"
    )

    # Dataset manifest only covers clean_file1
    manifests_dir = tmp_path / "datasets" / "manifests"
    manifests_dir.mkdir(parents=True)
    ds_manifest = manifests_dir / "krio-train-v1.manifest.json"
    ds_manifest.write_text(
        json.dumps({
            "schema_version": 2,
            "language": "krio",
            "split": "train",
            "version": 1,
            "files": {rel_p1: sha1},
            "eligible_counts": {"text_training": 10},
            "skipped": {},
            "built_at": "2026-11-12T00:00:00Z",
        }),
        encoding="utf-8",
    )

    licence_review = {
        "reviewer": "reviewer1",
        "reviewed_on": "2026-11-12",
        "licence": "Apache-2.0",
        "commercial_permitted": True,
        "intended_use": "research",
    }

    # Approval manifest references both file1 and file2
    approval_path = tmp_path / "approval_uncovered.json"
    approval_path.write_text(
        json.dumps({
            "run_id": "run_uncovered",
            "task": "chat_llm_sft",
            "base_model_id": "Qwen/Qwen2.5-3B",
            "base_model_revision": "abc1234",
            "model_licence_review": licence_review,
            "dataset_versions": {"train": "0.1.0"},
            "dataset_sha256_hashes": {rel_p1: sha1, rel_p2: sha2},
            "dataset_manifest_id": "krio-train-v1",
            "split_manifest_id": "speaker_splits_v1",
            "benchmark_id": "bench_v1",
            "estimated_gpu_hours": 1.0,
            "estimated_cost_usd": 2.0,
            "approver": "owner",
        }),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="not covered by dataset manifest"):
        verify_training_manifest(approval_path, base_dir=tmp_path)


def test_training_guard_rejects_task_mismatch(tmp_path: Path, monkeypatch):
    """PA-010: Approval manifest task != requested task_key must raise RuntimeError."""
    import sierraleone_ai.config as cfg_module
    from sierraleone_ai.training.guard import require_training_approval

    monkeypatch.setattr(cfg_module.settings, "allow_training", True)
    monkeypatch.setattr(cfg_module.settings, "data_dir", tmp_path)
    monkeypatch.setattr(cfg_module.settings, "models_dir", tmp_path / "models")

    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True)
    (models_dir / "selected_base_model.json").write_text(
        json.dumps({
            "hf_repo": "Qwen/Qwen2.5-3B",
            "tokenizer_audit_completed": True,
            "native_benchmark_completed": True,
        }),
        encoding="utf-8",
    )

    krio_dir = tmp_path / "krio"
    clean_file = krio_dir / "validated" / "krio_text_train.jsonl"
    clean_file.parent.mkdir(parents=True)
    clean_file.write_text('{"text": "val"}\n', encoding="utf-8")
    real_sha = hashlib.sha256(clean_file.read_bytes()).hexdigest()

    splits_dir = krio_dir / "splits"
    splits_dir.mkdir(parents=True)
    (splits_dir / "speaker_splits_v1.json").write_text(
        json.dumps({"speaker_to_split": {}}), encoding="utf-8"
    )

    manifests_dir = tmp_path / "datasets" / "manifests"
    manifests_dir.mkdir(parents=True)
    ds_manifest = manifests_dir / "krio-train-v1.manifest.json"
    rel_clean_path = str(clean_file.relative_to(tmp_path)).replace("\\", "/")
    ds_manifest.write_text(
        json.dumps({
            "schema_version": 2,
            "language": "krio",
            "split": "train",
            "version": 1,
            "files": {rel_clean_path: real_sha},
            "eligible_counts": {"text_training": 100000},
            "skipped": {},
            "built_at": "2026-11-12T00:00:00Z",
        }),
        encoding="utf-8",
    )

    licence_review = {
        "reviewer": "reviewer1",
        "reviewed_on": "2026-11-12",
        "licence": "Apache-2.0",
        "commercial_permitted": True,
        "intended_use": "research",
    }
    approval_path = tmp_path / "approval.json"
    approval_path.write_text(
        json.dumps({
            "run_id": "run_mismatch_task",
            "task": "chat_llm_cpt",
            "base_model_id": "Qwen/Qwen2.5-3B",
            "base_model_revision": "abc1234",
            "model_licence_review": licence_review,
            "dataset_versions": {"train": "0.1.0"},
            "dataset_sha256_hashes": {rel_clean_path: real_sha},
            "dataset_manifest_id": "krio-train-v1",
            "split_manifest_id": "speaker_splits_v1",
            "benchmark_id": "bench_v1",
            "estimated_gpu_hours": 2.0,
            "estimated_cost_usd": 4.0,
            "approver": "project_owner",
        }),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="does not match requested task"):
        require_training_approval(
            "test_task",
            language="krio",
            manifest_path=approval_path,
            task_key="chat_llm_sft",
            base_dir=tmp_path,
            check_readiness=False,
        )

