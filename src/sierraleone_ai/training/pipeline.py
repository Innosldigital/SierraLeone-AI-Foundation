"""
Krio & Multilingual Language Model Training Pipeline Infrastructure (Phase 4).

Prepares configurations and checkpoint/versioning protocols for:
- Continued pretraining (CPT)
- Instruction fine-tuning (SFT)
- LoRA / QLoRA parameter-efficient experiments
- Model checkpointing & semantic model versioning

All execution entry points enforce `require_training_approval()` so no expensive
GPU training can start without explicit human review of datasets and base models.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any

from sierraleone_ai.training.guard import require_training_approval


class AdaptationMethod(str, Enum):
    FULL_FINETUNE = "full"
    LORA = "lora"
    QLORA_4BIT = "qlora_4bit"
    QLORA_8BIT = "qlora_8bit"


@dataclass
class LoRAConfig:
    """Hyperparameters for LoRA / QLoRA adaptation."""

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    bias: str = "none"
    # Quantisation is controlled by AdaptationMethod (qlora_4bit / qlora_8bit).
    # Do not duplicate that decision here with a boolean flag.
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"  # safer default across GPU generations
    modules_to_save: list[str] = field(
        default_factory=list
    )
    # e.g. ["embed_tokens", "lm_head"] if the vocabulary has been extended with Krio tokens.
    # Leave empty when no vocabulary extension is performed.


@dataclass
class CheckpointPolicy:
    """Policy for saving and retaining model checkpoints."""

    output_dir: Path = Path("models/krio-llm/checkpoints")
    save_strategy: str = "steps"
    save_steps: int = 250
    save_total_limit: int = 3
    eval_steps: int = 250
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    early_stopping_patience: int = 3  # Stop if eval_loss does not improve for N evals


@dataclass
class ModelVersionMetadata:
    """Semantic versioning and provenance metadata for a trained model artifact."""

    model_name: str = "cafp-krio-ai"
    version: str = "0.1.0-alpha"
    language: str = "krio"
    base_model_hf_repo: str = ""
    dataset_version: str = "0.1.0"
    adaptation_method: AdaptationMethod = AdaptationMethod.QLORA_4BIT
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    native_speaker_evaluated: bool = False
    approved_for_deployment: bool = False


@dataclass
class ContinuedPretrainingConfig:
    """Configuration for Phase 4A: Continued Pretraining on verified Krio text."""

    base_model_repo: str
    language: str = "krio"
    # Uses validated/ not cleaned/ — training requires ≥2 native-speaker reviews
    train_data_path: Path = Path("datasets/krio/validated/krio_text_train.jsonl")
    val_data_path: Path = Path("datasets/krio/validated/krio_text_validation.jsonl")
    max_seq_length: int = 2048
    learning_rate: float = 2e-5
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.05
    adaptation_method: AdaptationMethod = AdaptationMethod.QLORA_4BIT
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    # CPT gets its own output directory, separate from SFT checkpoints
    checkpoint: CheckpointPolicy = field(
        default_factory=lambda: CheckpointPolicy(
            output_dir=Path("models/krio-llm/checkpoints-cpt")
        )
    )

    @classmethod
    def from_json(cls, path: str | Path) -> ContinuedPretrainingConfig:
        """Load a ContinuedPretrainingConfig from a JSON file."""
        import dataclasses

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        base_model_repo = data.pop("base_model_repo")
        # Coerce Path fields
        for key in ("train_data_path", "val_data_path"):
            if key in data:
                data[key] = Path(data[key])
        # Coerce nested dataclasses
        if "lora" in data:
            data["lora"] = LoRAConfig(**data["lora"])
        cp = data.pop("checkpoint", None) or data.pop("checkpoint_policy", None)
        if cp:
            if "output_dir" in cp:
                cp["output_dir"] = Path(cp["output_dir"])
            data["checkpoint"] = CheckpointPolicy(**cp)
        if "adaptation_method" in data:
            data["adaptation_method"] = AdaptationMethod(data["adaptation_method"])
        known = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(base_model_repo=base_model_repo, **filtered)


@dataclass
class InstructionFineTuningConfig:
    """Configuration for Phase 4B: Instruction Fine-Tuning on Krio conversations, QA, and translations."""

    pretrained_checkpoint: Path
    language: str = "krio"
    conversations_path: Path = Path("datasets/krio/conversations/conversations_train.jsonl")
    translations_path: Path = Path("datasets/krio/translations/translations_english_krio_train.jsonl")
    max_seq_length: int = 2048
    learning_rate: float = 1e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    adaptation_method: AdaptationMethod = AdaptationMethod.QLORA_4BIT
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    # SFT gets its own output directory, separate from CPT checkpoints
    checkpoint: CheckpointPolicy = field(
        default_factory=lambda: CheckpointPolicy(
            output_dir=Path("models/krio-llm/checkpoints-sft")
        )
    )

    @classmethod
    def from_json(cls, path: str | Path) -> InstructionFineTuningConfig:
        """Load an InstructionFineTuningConfig from a JSON file."""
        import dataclasses

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        pretrained_checkpoint = Path(data.pop("pretrained_checkpoint"))
        # Coerce Path fields
        for key in ("conversations_path", "translations_path"):
            if key in data:
                data[key] = Path(data[key])
        # Coerce nested dataclasses
        if "lora" in data:
            data["lora"] = LoRAConfig(**data["lora"])
        cp = data.pop("checkpoint", None) or data.pop("checkpoint_policy", None)
        if cp:
            if "output_dir" in cp:
                cp["output_dir"] = Path(cp["output_dir"])
            data["checkpoint"] = CheckpointPolicy(**cp)
        if "adaptation_method" in data:
            data["adaptation_method"] = AdaptationMethod(data["adaptation_method"])
        known = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(pretrained_checkpoint=pretrained_checkpoint, **filtered)


def _assert_not_raw_path(path: Path, field_name: str) -> None:
    parts = PurePosixPath(str(path).replace("\\", "/")).parts
    if "raw" in parts:
        raise ValueError(
            f"{field_name} ('{path}') points inside a 'raw/' directory. "
            "No training code may read from raw/ folders directly."
        )


def run_continued_pretraining(
    config: ContinuedPretrainingConfig,
    manifest_path: Path | None = None,
    *,
    dry_run: bool = False,
) -> ModelVersionMetadata | dict[str, Any]:
    """Guarded entry point for continued pretraining."""
    if config.base_model_repo.strip().upper().startswith("TBD"):
        raise ValueError(
            f"base_model_repo is still unresolved ('{config.base_model_repo}'). "
            "Select a verified base model before running continued pretraining."
        )
    _assert_not_raw_path(config.train_data_path, "train_data_path")
    _assert_not_raw_path(config.val_data_path, "val_data_path")
    plan = require_training_approval(
        task=f"Continued pretraining ({config.language}) on {config.base_model_repo}",
        language=config.language,
        manifest_path=manifest_path,
        task_key="chat_llm_cpt",
        config=config,
        dry_run=dry_run,
    )
    if dry_run:
        return plan
    raise NotImplementedError(
        "Continued pretraining requires native-speaker-validated corpus in "
        f"{config.train_data_path} and approved base model weights."
    )


def run_instruction_finetuning(
    config: InstructionFineTuningConfig,
    manifest_path: Path | None = None,
    *,
    dry_run: bool = False,
) -> ModelVersionMetadata | dict[str, Any]:
    """Guarded entry point for instruction fine-tuning."""
    _assert_not_raw_path(config.conversations_path, "conversations_path")
    _assert_not_raw_path(config.translations_path, "translations_path")
    plan = require_training_approval(
        task=f"Instruction fine-tuning ({config.language})",
        language=config.language,
        manifest_path=manifest_path,
        task_key="chat_llm_sft",
        config=config,
        dry_run=dry_run,
    )
    if dry_run:
        return plan
    raise NotImplementedError(
        "Instruction fine-tuning requires a completed continued-pretraining checkpoint "
        "and native-speaker-validated conversation/translation datasets."
    )
