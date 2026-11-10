"""
Text-to-Speech (TTS) infrastructure for Sierra Leonean languages.

Provides:
- Studio quality readiness checks for TTS training audio (SNR, sample rate, consent)
- TTS model configuration stubs
- Guarded TTS training entry point
"""
from __future__ import annotations

from dataclasses import dataclass

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
    elif record.recording_environment != cfg.required_environment:
        issues.append(
            f"Recording environment '{record.recording_environment}' != '{cfg.required_environment}'"
        )

    if record.duration_seconds is None:
        issues.append("duration not measured – required for TTS quality gate")

    return (len(issues) == 0, issues)


def start_tts_training(config: TTSTrainingConfig | None = None) -> None:
    """Guarded entry point for TTS model training."""
    require_training_approval(task="Krio TTS model training")
    raise NotImplementedError(
        "TTS training requires studio-grade verified recordings and explicit speaker consent."
    )

