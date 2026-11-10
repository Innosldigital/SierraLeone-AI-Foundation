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


@dataclass
class ASRTrainingConfig:
    """Configuration for fine-tuning an open-weight ASR model on Krio or other SL languages."""

    language: str = "krio"
    base_model: str = "openai/whisper-small"
    target_sample_rate_hz: int = 16000
    min_duration_seconds: float = 0.5
    max_duration_seconds: float = 30.0
    require_dual_verified_transcripts: bool = True
    learning_rate: float = 1e-5
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    max_steps: int = 4000
    output_dir: str = "models/krio-asr/checkpoints"


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
    def _extract_speakers(records: list[AudioRecord | VerifiedSpeechTranscriptionRecord]) -> set[str]:
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
        dry_run=dry_run,
    )
    if dry_run:
        return plan
    raise NotImplementedError(
        f"ASR training for {cfg.language} ({cfg.base_model}) requires verified audio "
        "recordings in datasets/krio/audio/ and GPU hardware approval."
    )

