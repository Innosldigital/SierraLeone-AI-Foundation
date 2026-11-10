"""
Model registry for the Sierra Leone AI Foundation (SLAI-005 / SLAI-032).

Tracks candidate base models for adaptation, separated by task type
(`chat_llm`, `translation_seq2seq`, `asr`, `tts`) so causal LMs, encoder-decoders,
speech recognisers, and speech synthesisers are never compared as equals.

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
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

from sierraleone_ai.config import settings


class ModelTask(str, Enum):
    CHAT_LLM = "chat_llm"
    TRANSLATION_SEQ2SEQ = "translation_seq2seq"
    ASR = "asr"
    TTS = "tts"


TaskType = Literal["chat_llm", "translation_seq2seq", "asr", "tts"]


@dataclass
class CandidateModel:
    """Metadata for a candidate base model. All fields must be verified from official sources."""

    name: str
    hf_repo: str
    parameters: str
    licence: str                    # "VERIFY on model card" until confirmed
    multilingual: bool
    african_language_support: str   # factual description, never a headline marketing claim
    krio_tokenizer_coverage: str    # always "unknown – needs measurement" until measured
    min_vram_gb: int
    notes: str
    task: TaskType = "chat_llm"
    recommended: bool = False
    verified_on: str | None = None   # ISO date string when entry was last verified
    source_url: str | None = None    # URL to model card or technical report


# ---------------------------------------------------------------------------
# Candidate models for Sierra Leonean language adaptation (SLAI-005 / SLAI-032)
# ---------------------------------------------------------------------------

CANDIDATE_MODELS: list[CandidateModel] = [
    # --- Causal / Chat LLMs (task="chat_llm") ---
    CandidateModel(
        name="Qwen3-4B",
        hf_repo="Qwen/Qwen3-4B",
        parameters="4B",
        licence="VERIFY on model card",
        multilingual=True,
        african_language_support=(
            "Multilingual pretraining reported on model card, but Krio (kri) support is "
            "UNVERIFIED. Do not treat headline language counts as Krio coverage."
        ),
        krio_tokenizer_coverage="unknown – needs measurement",
        min_vram_gb=8,
        notes="Instruction/chat variant of Qwen3-4B; requires tokenizer fertility audit and benchmark.",
        task="chat_llm",
        recommended=False,
        verified_on=None,
        source_url="https://huggingface.co/Qwen/Qwen3-4B",
    ),
    CandidateModel(
        name="Qwen3-4B-Base",
        hf_repo="Qwen/Qwen3-4B-Base",
        parameters="4B",
        licence="VERIFY on model card",
        multilingual=True,
        african_language_support=(
            "Used in AfriqueLLM (arXiv 2601.06395, ACL 2026). "
            "Krio not among AfriqueLLM languages; Qwen3 Krio coverage is UNVERIFIED."
        ),
        krio_tokenizer_coverage="unknown – needs measurement",
        min_vram_gb=8,
        notes="Source: arXiv 2601.06395. Pretrained base checkpoint; verify Krio tokeniser efficiency.",
        task="chat_llm",
        recommended=False,
        verified_on=None,
        source_url="https://huggingface.co/Qwen/Qwen3-4B-Base",
    ),
    CandidateModel(
        name="Qwen3-8B-Base",
        hf_repo="Qwen/Qwen3-8B-Base",
        parameters="8B",
        licence="VERIFY on model card",
        multilingual=True,
        african_language_support=(
            "Used in AfriqueLLM (arXiv 2601.06395, ACL 2026). "
            "Krio not among AfriqueLLM languages; Qwen3 Krio coverage is UNVERIFIED."
        ),
        krio_tokenizer_coverage="unknown – needs measurement",
        min_vram_gb=16,
        notes="Source: arXiv 2601.06395. Larger Qwen3 variant.",
        task="chat_llm",
        recommended=False,
        verified_on=None,
        source_url="https://huggingface.co/Qwen/Qwen3-8B-Base",
    ),
    CandidateModel(
        name="Gemma-3 1B",
        hf_repo="google/gemma-3-1b-pt",
        parameters="1B",
        licence="VERIFY on model card (Gemma Terms of Use)",
        multilingual=True,
        african_language_support=(
            "Compact Gemma-3 pretrained checkpoint. Krio is NOT officially supported; "
            "do not treat headline multilingual claims as Krio support."
        ),
        krio_tokenizer_coverage="unknown – needs measurement",
        min_vram_gb=4,
        notes="Lightweight 1B candidate suitable for low-VRAM QLoRA experimentation.",
        task="chat_llm",
        recommended=False,
        verified_on=None,
        source_url="https://huggingface.co/google/gemma-3-1b-pt",
    ),
    CandidateModel(
        name="Gemma-3 4B",
        hf_repo="google/gemma-3-4b-pt",
        parameters="4B",
        licence="VERIFY on model card (Gemma Terms of Use)",
        multilingual=False,
        african_language_support=(
            "Used in AfriqueLLM (arXiv 2601.06395, ACL 2026) for African language adaptation. "
            "Krio not among the AfriqueLLM languages; base model has no verified Krio data."
        ),
        krio_tokenizer_coverage="unknown – needs measurement",
        min_vram_gb=8,
        notes=(
            "Smaller scale; suitable for resource-constrained training. "
            "Verify Gemma Terms of Use for research vs commercial use."
        ),
        task="chat_llm",
        recommended=False,
        verified_on=None,
        source_url="https://huggingface.co/google/gemma-3-4b-pt",
    ),
    CandidateModel(
        name="Gemma-3 12B",
        hf_repo="google/gemma-3-12b-pt",
        parameters="12B",
        licence="VERIFY on model card (Gemma Terms of Use)",
        multilingual=False,
        african_language_support=(
            "Used in AfriqueLLM (arXiv 2601.06395, ACL 2026) for African language adaptation. "
            "Krio not among the AfriqueLLM languages; base model has no verified Krio data."
        ),
        krio_tokenizer_coverage="unknown – needs measurement",
        min_vram_gb=24,
        notes="Larger Gemma-3 variant. Requires 24 GB VRAM for QLoRA. Source: arXiv 2601.06395.",
        task="chat_llm",
        recommended=False,
        verified_on=None,
        source_url="https://huggingface.co/google/gemma-3-12b-pt",
    ),
    CandidateModel(
        name="Llama-3.2 1B",
        hf_repo="meta-llama/Llama-3.2-1B",
        parameters="1B",
        licence="VERIFY on model card (Llama 3.2 Community Licence)",
        multilingual=True,
        african_language_support=(
            "Krio is NOT among Llama 3.2's officially supported languages "
            "(English, German, French, Italian, Portuguese, Hindi, Spanish, Thai)."
        ),
        krio_tokenizer_coverage="unknown – needs measurement",
        min_vram_gb=4,
        notes="Compact 1B baseline; English-lexified transfer to Krio must be measured empirically.",
        task="chat_llm",
        recommended=False,
        verified_on=None,
        source_url="https://huggingface.co/meta-llama/Llama-3.2-1B",
    ),
    CandidateModel(
        name="Llama-3.2 3B",
        hf_repo="meta-llama/Llama-3.2-3B",
        parameters="3B",
        licence="VERIFY on model card (Llama 3.2 Community Licence)",
        multilingual=True,
        african_language_support=(
            "Krio is NOT among Llama 3.2's officially supported languages "
            "(English, German, French, Italian, Portuguese, Hindi, Spanish, Thai)."
        ),
        krio_tokenizer_coverage="unknown – needs measurement",
        min_vram_gb=8,
        notes="3B baseline candidate for LoRA/QLoRA adaptation; requires custom Krio evaluation.",
        task="chat_llm",
        recommended=False,
        verified_on=None,
        source_url="https://huggingface.co/meta-llama/Llama-3.2-3B",
    ),
    CandidateModel(
        name="Llama-3.1-8B",
        hf_repo="meta-llama/Llama-3.1-8B",
        parameters="8B",
        licence="VERIFY on model card (Llama-3.1 Community Licence)",
        multilingual=False,
        african_language_support=(
            "Primarily English-focused. Used as the base for AfriqueLLM (arXiv 2601.06395, ACL 2026) "
            "which performed continued pretraining across 20 African languages (~26B tokens). "
            "Krio not among the AfriqueLLM languages; plain Llama-3.1-8B has no Krio data."
        ),
        krio_tokenizer_coverage="unknown – needs measurement; English-centric tokeniser may fragment Krio",
        min_vram_gb=16,
        notes=(
            "Strong baseline; widely tested for African language adaptation (AfriqueLLM 2026). "
            "QLoRA at 4-bit runs on 16 GB VRAM. Tokeniser fertility must be measured first."
        ),
        task="chat_llm",
        recommended=False,
        verified_on=None,
        source_url="https://huggingface.co/meta-llama/Llama-3.1-8B",
    ),
    CandidateModel(
        name="Aya-23 8B",
        hf_repo="CohereForAI/aya-23-8B",
        parameters="8B",
        licence="VERIFY on model card (CC-BY-NC-4.0)",
        multilingual=True,
        african_language_support=(
            "No sub-Saharan African languages. The 23 languages are: Arabic, Chinese "
            "(Simplified/Traditional), Czech, Dutch, English, French, German, Greek, "
            "Hebrew, Hindi, Indonesian, Italian, Japanese, Korean, Persian, Polish, "
            "Portuguese, Romanian, Russian, Spanish, Turkish, Ukrainian, Vietnamese. "
            "Arabic is the only African-spoken language. Krio is NOT covered."
        ),
        krio_tokenizer_coverage="unknown – needs measurement",
        min_vram_gb=16,
        notes=(
            "Despite the name, Aya-23 does not cover sub-Saharan Africa. "
            "CC-BY-NC-4.0 licence restricts commercial use."
        ),
        task="chat_llm",
        recommended=False,
        verified_on=None,
        source_url="https://huggingface.co/CohereForAI/aya-23-8B",
    ),
    CandidateModel(
        name="Tiny Aya (Aya-Expanse variants)",
        hf_repo="CohereForAI/aya-expanse-8b",
        parameters="~3.35B–8B (multiple variants)",
        licence="VERIFY on model card",
        multilingual=True,
        african_language_support=(
            "Reported as 70+ languages; 'Earth' and 'Global' variants target African languages. "
            "Krio (kri) coverage is UNVERIFIED — check model card before use."
        ),
        krio_tokenizer_coverage="unknown – needs measurement",
        min_vram_gb=8,
        notes=(
            "Released Feb 2026. Smaller scale (3.35B) makes QLoRA more feasible. "
            "Licence and exact language list must be verified from model card."
        ),
        task="chat_llm",
        recommended=False,
        verified_on=None,
        source_url="https://huggingface.co/CohereForAI/aya-expanse-8b",
    ),
    CandidateModel(
        name="AfroLLaMA / AfroLlama_V1",
        hf_repo="Jacaranda/AfroLlama_V1",
        parameters="VERIFY on model card (reported ~8B, Llama-3-based)",
        licence="VERIFY on model card",
        multilingual=True,
        african_language_support=(
            "Reported as African-language LoRA continued-pretraining + instruction-tuning on "
            "~6 African languages. Krio (kri) is NOT listed in published descriptions. "
            "Verify language list from model card before use."
        ),
        krio_tokenizer_coverage="unknown – Krio not in reported training languages",
        min_vram_gb=16,
        notes=(
            "Entry requires manual verification of: (1) exact base model (Llama-2 or Llama-3), "
            "(2) parameter count, (3) licence, (4) exact language list."
        ),
        task="chat_llm",
        recommended=False,
        verified_on=None,
        source_url="https://huggingface.co/Jacaranda/AfroLlama_V1",
    ),

    # --- Encoder-Decoder Translation Candidates (task="translation_seq2seq") ---
    CandidateModel(
        name="mT5-base",
        hf_repo="google/mt5-base",
        parameters="580M",
        licence="VERIFY on model card (Apache-2.0)",
        multilingual=True,
        african_language_support=(
            "Trained on mC4 corpus; Krio (kri) is NOT explicitly listed and coverage is unverified."
        ),
        krio_tokenizer_coverage="unknown – needs measurement",
        min_vram_gb=4,
        notes="Encoder-decoder architecture for sequence-to-sequence translation experiments only.",
        task="translation_seq2seq",
        recommended=False,
        verified_on=None,
        source_url="https://huggingface.co/google/mt5-base",
    ),
    CandidateModel(
        name="NLLB-200 Distilled 600M",
        hf_repo="facebook/nllb-200-distilled-600M",
        parameters="600M",
        licence="VERIFY on model card (CC-BY-NC-4.0)",
        multilingual=True,
        african_language_support=(
            "Includes Krio (`kri_Latn`) in the NLLB-200 language inventory; quality on Sierra Leonean "
            "Krio requires independent evaluation on a native-reviewed held-out benchmark."
        ),
        krio_tokenizer_coverage="unknown – needs measurement",
        min_vram_gb=4,
        notes="Encoder-decoder translation baseline; check non-commercial licence constraints.",
        task="translation_seq2seq",
        recommended=False,
        verified_on=None,
        source_url="https://huggingface.co/facebook/nllb-200-distilled-600M",
    ),

    # --- Automatic Speech Recognition Candidates (task="asr") ---
    CandidateModel(
        name="Whisper Small",
        hf_repo="openai/whisper-small",
        parameters="244M",
        licence="VERIFY on model card (MIT)",
        multilingual=True,
        african_language_support=(
            "Krio (`kri`) is NOT in OpenAI Whisper's pretrained language list; requires "
            "supervised fine-tuning on verified Sierra Leonean Krio speech."
        ),
        krio_tokenizer_coverage="unknown – needs measurement",
        min_vram_gb=4,
        notes="ASR baseline architecture for comparison against w2v-BERT / MMS models.",
        task="asr",
        recommended=False,
        verified_on=None,
        source_url="https://huggingface.co/openai/whisper-small",
    ),
    CandidateModel(
        name="DONDO w2v-BERT Sierra Leone ASR",
        hf_repo="KhayaAI/w2v-bert-kri-men-tem-en",
        parameters="605M",
        licence="VERIFY on model card (Apache-2.0)",
        multilingual=True,
        african_language_support=(
            "Multilingual w2v-BERT 2.0 CTC checkpoint covering Krio (kri), Mende (men), "
            "Temne (tem), and English (en), trained primarily on read religious speech. "
            "Out-of-domain conversational and agricultural WER is unverified."
        ),
        krio_tokenizer_coverage="unknown – character/CTC vocabulary",
        min_vram_gb=8,
        notes="Candidate Sierra Leone ASR model (arXiv:2607.21540); requires independent benchmark evaluation.",
        task="asr",
        recommended=False,
        verified_on=None,
        source_url="https://huggingface.co/KhayaAI/w2v-bert-kri-men-tem-en",
    ),

    # --- Text-to-Speech Candidates (task="tts") ---
    CandidateModel(
        name="MMS-TTS Baseline",
        hf_repo="facebook/mms-tts",
        parameters="VITS (~36M per language)",
        licence="VERIFY on model card (CC-BY-NC-4.0)",
        multilingual=True,
        african_language_support=(
            "Meta MMS covers 1,100+ languages primarily from religious recordings; "
            "Sierra Leonean Krio naturalness and licence terms must be verified."
        ),
        krio_tokenizer_coverage="unknown – phoneme/character VITS input",
        min_vram_gb=4,
        notes="TTS architecture candidate; requires studio-grade consented speech before adaptation.",
        task="tts",
        recommended=False,
        verified_on=None,
        source_url="https://huggingface.co/facebook/mms-tts",
    ),
]


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
    for m in CANDIDATE_MODELS:
        if m.hf_repo == hf_repo and (task is None or m.task == task):
            return m
    return None


def list_candidates(task: TaskType | None = None) -> list[CandidateModel]:
    """Return candidate models, optionally filtered by task (`chat_llm`, `translation_seq2seq`, `asr`, `tts`)."""
    if task is None:
        return list(CANDIDATE_MODELS)
    return [m for m in CANDIDATE_MODELS if m.task == task]


def get_all_candidates(task: TaskType | None = None) -> list[CandidateModel]:
    """Return all registered candidate models. Alias for list_candidates()."""
    return list_candidates(task=task)
