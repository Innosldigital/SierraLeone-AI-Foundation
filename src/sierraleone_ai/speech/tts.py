"""
Text-to-Speech (TTS) infrastructure for Sierra Leonean languages.

Provides:
- Studio quality readiness checks for TTS training audio (SNR, sample rate, consent)
- TTS model configuration stubs
- Guarded TTS training entry point
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sierraleone_ai.datasets.schema import VerifiedSpeechTranscriptionRecord
from sierraleone_ai.training.guard import require_training_approval


@dataclass
class TTSTrainingConfig:
    """Configuration for training/adapting a TTS model for Sierra Leonean languages."""

    language: str = "krio"
    base_architecture: str = "mms-tts-or-vits"
    min_sample_rate_hz: int = 22050
    min_snr_db: float = 25.0
    required_environment: str = "studio"
    output_dir: str = "models/krio-tts/checkpoints"


def check_tts_recording_quality(
    record: VerifiedSpeechTranscriptionRecord,
    config: TTSTrainingConfig | None = None,
    dataset_root: Path | None = None,
) -> tuple[bool, list[str]]:
    """
    Validate whether a verified speech recording meets studio TTS quality standards.

    Returns:
        Tuple of (is_eligible, list_of_reasons_if_ineligible)
    """
    cfg = config or TTSTrainingConfig()
    issues: list[str] = []

    if not record.metadata.contributor_consent:
        issues.append("Missing explicit contributor consent for voice synthesis")
    elif "voice_synthesis" not in record.metadata.consent_scope:
        issues.append(
            "Missing explicit 'voice_synthesis' in consent_scope – ASR/general consent cannot be inferred for TTS"
        )

    if record.sample_rate_hz is None or record.sample_rate_hz < cfg.min_sample_rate_hz:
        issues.append(
            f"Sample rate {record.sample_rate_hz} Hz is below minimum {cfg.min_sample_rate_hz} Hz"
        )

    if record.snr_db is None:
        issues.append("SNR not measured – required for studio TTS quality gate")
    elif record.snr_db < cfg.min_snr_db:
        issues.append(f"SNR {record.snr_db} dB is below minimum {cfg.min_snr_db} dB")

    if record.recording_environment is None:
        issues.append("recording environment not recorded – required for TTS quality gate")
    elif (
        record.recording_environment.strip().casefold()
        != cfg.required_environment.strip().casefold()
    ):
        issues.append(
            f"Recording environment '{record.recording_environment}' != '{cfg.required_environment}'"
        )

    if record.duration_seconds is None:
        issues.append("duration not measured – required for TTS quality gate")

    if dataset_root is not None:
        try:
            record.verify_audio_file(dataset_root)
        except (FileNotFoundError, ValueError) as exc:
            issues.append(f"Audio file verification failed: {exc}")

    return (len(issues) == 0, issues)


def start_tts_training(
    config: TTSTrainingConfig | None = None,
    manifest_path: Path | None = None,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Guarded entry point for TTS model training."""
    cfg = config or TTSTrainingConfig()
    plan = require_training_approval(
        task=f"{cfg.language} TTS model training ({cfg.base_architecture})",
        language=cfg.language,
        manifest_path=manifest_path,
        task_key="tts_training",
        config=cfg,
        dry_run=dry_run,
    )
    if dry_run:
        return plan
    raise NotImplementedError(
        f"TTS training for {cfg.language} ({cfg.base_architecture}) requires studio-grade "
        "verified recordings and explicit speaker consent."
    )
