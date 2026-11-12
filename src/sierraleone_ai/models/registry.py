"""
Model registry for the Sierra Leone AI Foundation (SLAI-005 / SLAI-032 / PA-005).

Tracks candidate base models for adaptation, separated by task type
(`chat_llm`, `translation_seq2seq`, `asr`, `tts`) so causal LMs, encoder-decoders,
speech recognisers, and speech synthesisers are never compared as equals.

Candidate data is stored in `models/candidates.json` and validated by Pydantic.
No model is downloaded here — this is metadata only.
Download and training require all prerequisites in training/guard.py.

IMPORTANT:
  - All entries remain `recommended=False` until both the tokenisation audit
    and native-reviewed Krio benchmark are completed.
  - `get_recommended()` reads a formal decision file (`models/selected_base_model.json`),
    not a hard-coded flag.
  - Headline marketing claims such as "100+ languages" must never be recorded as Krio support.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from sierraleone_ai.config import settings


class ModelTask(str, Enum):
    CHAT_LLM = "chat_llm"
    TRANSLATION_SEQ2SEQ = "translation_seq2seq"
    ASR = "asr"
    TTS = "tts"


TaskType = Literal["chat_llm", "translation_seq2seq", "asr", "tts"]
KrioSupport = Literal["documented", "undocumented", "unsupported", "unknown"]


class CandidateModel(BaseModel):
    """Metadata for a candidate base model. All fields must be verified from official sources (PA-005)."""

    name: str = Field(..., min_length=1)
    hf_repo: str = Field(..., min_length=1)
    parameters: str = Field(..., min_length=1)
    licence: str = Field(..., min_length=1)
    multilingual: bool | None = None
    african_language_support: str = Field(..., min_length=1)
    krio_support: KrioSupport = "unknown"
    krio_tokenizer_coverage: str = "unknown – needs measurement"
    min_vram_gb: int = Field(..., ge=1)
    notes: str = ""
    task: TaskType = "chat_llm"
    recommended: bool = False
    verified: bool = False
    verified_on: str | None = None
    source_url: str = Field(..., min_length=1)
    evidence_url: str | None = None

    @model_validator(mode="after")
    def validate_verified_and_evidence(self) -> CandidateModel:
        if self.verified and not self.verified_on:
            raise ValueError(
                f"Candidate '{self.name}' is marked verified=True but missing verified_on date"
            )
        if self.krio_support == "documented" and not self.evidence_url:
            raise ValueError(
                f"Candidate '{self.name}' claims krio_support='documented' but missing evidence_url"
            )
        return self


def load_candidate_models(path: Path | None = None) -> list[CandidateModel]:
    candidate_file = path or (settings.models_dir / "candidates.json")
    if not candidate_file.is_file():
        fallback = settings.project_root / "models" / "candidates.json"
        if fallback.is_file():
            candidate_file = fallback
        else:
            return []
    data = json.loads(candidate_file.read_text(encoding="utf-8-sig"))
    return [CandidateModel.model_validate(item) for item in data]


CANDIDATE_MODELS: list[CandidateModel] = load_candidate_models()


def get_recommended(
    task: TaskType | None = None,
    decision_file: Path | None = None,
) -> CandidateModel | None:
    """
    Return the selected base model from `models/selected_base_model.json` if and only if
    a formal decision file exists and records both `tokenizer_audit_completed=True`
    and `native_benchmark_completed=True` (SLAI-032).

    Never relies on a hard-coded boolean flag in code. Returns None during the research phase.
    """
    target = decision_file or (settings.models_dir / "selected_base_model.json")
    if not target.exists():
        return None
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not (data.get("tokenizer_audit_completed") and data.get("native_benchmark_completed")):
        return None
    hf_repo = data.get("hf_repo")
    candidates = load_candidate_models() or CANDIDATE_MODELS
    for m in candidates:
        if m.hf_repo == hf_repo and (task is None or m.task == task):
            return m
    return None


def list_candidates(task: TaskType | None = None) -> list[CandidateModel]:
    """Return candidate models, optionally filtered by task (`chat_llm`, `translation_seq2seq`, `asr`, `tts`)."""
    candidates = load_candidate_models() or CANDIDATE_MODELS
    if task is None:
        return list(candidates)
    return [m for m in candidates if m.task == task]


def get_all_candidates(task: TaskType | None = None) -> list[CandidateModel]:
    """Return all registered candidate models. Alias for list_candidates()."""
    return list_candidates(task=task)
