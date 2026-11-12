"""
Automatic Speech Recognition (ASR) infrastructure for Sierra Leonean languages.

Provides:
- Audio dataset manifest validation for ASR training
- Speaker leakage audit across train/validation/test splits
- Candidate ASR base model specifications (Whisper, Meta MMS)
- Guarded ASR training configuration stub
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sierraleone_ai.datasets.schema import AudioRecord, VerifiedSpeechTranscriptionRecord
from sierraleone_ai.training.guard import require_training_approval


def get_default_asr_base_model() -> str:
    """Choose default ASR base model from registry decision or first candidate (PA-023)."""
    try:
        from sierraleone_ai.models.registry import get_recommended, list_candidates

        rec = get_recommended(task="asr")
        if rec is not None:
            return rec.hf_repo
        asr_candidates = list_candidates(task="asr")
        if asr_candidates:
            return asr_candidates[0].hf_repo
    except Exception:
        pass
    return "openai/whisper-small"


@dataclass
class ASRTrainingConfig:
    """Configuration for fine-tuning an open-weight ASR model on Krio or other SL languages."""

    language: str = "krio"
    base_model: str = field(default_factory=get_default_asr_base_model)
    target_sample_rate_hz: int = 16000
    min_duration_seconds: float = 0.5
    max_duration_seconds: float = 30.0
    require_dual_verified_transcripts: bool = True
    learning_rate: float = 1e-5
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    max_steps: int = 4000
    output_dir: str = "models/krio-asr/checkpoints"


def validate_asr_record(
    record: VerifiedSpeechTranscriptionRecord | AudioRecord,
    config: ASRTrainingConfig | None = None,
) -> tuple[bool, list[str]]:
    """
    Validate whether an audio record meets ASR training constraints (PA-023).
    Checks duration limits and dual verification if required.
    """
    cfg = config or ASRTrainingConfig()
    issues: list[str] = []

    if record.duration_seconds is None:
        issues.append("Duration not specified")
    else:
        if record.duration_seconds < cfg.min_duration_seconds:
            issues.append(
                f"Duration {record.duration_seconds:.2f}s is below minimum {cfg.min_duration_seconds}s"
            )
        if record.duration_seconds > cfg.max_duration_seconds:
            issues.append(
                f"Duration {record.duration_seconds:.2f}s exceeds maximum {cfg.max_duration_seconds}s"
            )

    if cfg.require_dual_verified_transcripts:
        meta = getattr(record, "metadata", None)
        reviews = getattr(meta, "reviews", None) if meta else None
        if reviews:
            approvals = [r for r in reviews if getattr(r, "verdict", None) == "approve"]
            if len(approvals) < 2:
                issues.append(f"Requires at least 2 native approvals (found {len(approvals)})")
        elif hasattr(record, "verified_by"):
            if len(getattr(record, "verified_by", [])) < 2:
                issues.append("Requires at least 2 verifiers in verified_by")
        else:
            issues.append("Record lacks verification reviews for dual-verification check")

    return len(issues) == 0, issues


@dataclass
class SpeakerLeakageReport:
    """Result of auditing an ASR dataset split for speaker overlap."""

    train_speakers: set[str] = field(default_factory=set)
    validation_speakers: set[str] = field(default_factory=set)
    test_speakers: set[str] = field(default_factory=set)
    leaked_train_val: set[str] = field(default_factory=set)
    leaked_train_test: set[str] = field(default_factory=set)
    leaked_val_test: set[str] = field(default_factory=set)

    @property
    def has_leakage(self) -> bool:
        return bool(self.leaked_train_val or self.leaked_train_test or self.leaked_val_test)


def audit_speaker_leakage(
    splits: dict[str, list[AudioRecord | VerifiedSpeechTranscriptionRecord]],
) -> SpeakerLeakageReport:
    """
    Verify that no speaker_id appears in more than one split (train, validation, test).
    Prevents speaker leakage from inflating ASR evaluation metrics.
    """

    def _extract_speakers(
        records: list[AudioRecord | VerifiedSpeechTranscriptionRecord],
    ) -> set[str]:
        return {r.speaker_id for r in records if r.speaker_id}

    train_spk = _extract_speakers(splits.get("train", []))
    val_spk = _extract_speakers(splits.get("validation", []))
    test_spk = _extract_speakers(splits.get("test", []))

    return SpeakerLeakageReport(
        train_speakers=train_spk,
        validation_speakers=val_spk,
        test_speakers=test_spk,
        leaked_train_val=train_spk & val_spk,
        leaked_train_test=train_spk & test_spk,
        leaked_val_test=val_spk & test_spk,
    )


def start_asr_training(
    config: ASRTrainingConfig | None = None,
    manifest_path: Path | None = None,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Guarded entry point for ASR fine-tuning."""
    cfg = config or ASRTrainingConfig()
    plan = require_training_approval(
        task=f"{cfg.language} ASR model training ({cfg.base_model})",
        language=cfg.language,
        manifest_path=manifest_path,
        task_key="asr_training",
        config=cfg,
        dry_run=dry_run,
    )
    if dry_run:
        return plan
    raise NotImplementedError(
        f"ASR training for {cfg.language} ({cfg.base_model}) requires verified audio "
        "recordings in datasets/krio/audio/ and GPU hardware approval."
    )
