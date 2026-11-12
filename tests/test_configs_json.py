"""
Tests for training JSON configs (SLAI-011 / SLAI-017).
"""

from __future__ import annotations

from pathlib import Path

from sierraleone_ai.training.pipeline import ContinuedPretrainingConfig, InstructionFineTuningConfig


def test_load_cpt_config():
    path = Path("training/continued-pretraining/cpt_config.json")
    assert path.exists()
    cfg = ContinuedPretrainingConfig.from_json(path)
    assert cfg.language == "krio"
    assert cfg.learning_rate == 2e-5
    assert cfg.checkpoint.output_dir == Path("models/krio-llm/checkpoints-cpt")


def test_load_sft_config():
    path = Path("training/fine-tuning/sft_lora_config.json")
    assert path.exists()
    cfg = InstructionFineTuningConfig.from_json(path)
    assert cfg.language == "krio"
    assert cfg.learning_rate == 1e-4
    assert cfg.checkpoint.output_dir == Path("models/krio-llm/checkpoints-sft")
    assert cfg.lora.bnb_4bit_compute_dtype == "float16"
