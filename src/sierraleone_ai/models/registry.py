"""
Model registry for the Sierra Leone AI Foundation.

Tracks candidate base models for adaptation.
No model is downloaded here — this is metadata only.
Download and training require all prerequisites in training/guard.py.

IMPORTANT: All entries marked recommended=False until tokenisation audit is complete.
           licence and african_language_support fields must be verified from the
           official model card before trusting — do not fill from memory.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CandidateModel:
    """Metadata for a candidate base model. All fields must be verified from official sources."""

    name: str
    hf_repo: str
    parameters: str
    licence: str                    # "VERIFY on model card" until confirmed
    multilingual: bool
    african_language_support: str   # factual description, not a rating
    krio_tokenizer_coverage: str    # always "unknown – needs measurement" until measured
    min_vram_gb: int
    notes: str
    recommended: bool = False
    verified_on: str | None = None   # ISO date string when entry was last verified
    source_url: str | None = None    # URL to model card or technical report


# ---------------------------------------------------------------------------
# Candidate models for Krio language adaptation
# (Research phase — not yet downloaded or trained)
#
# VERIFICATION STATUS:
#   Every entry below needs manual verification from the linked model card.
#   Fields marked "VERIFY" are placeholders.
# ---------------------------------------------------------------------------

CANDIDATE_MODELS: list[CandidateModel] = [

    CandidateModel(
        name="Aya-23 8B",
        hf_repo="CohereForAI/aya-23-8B",
        parameters="8B",
        licence="CC-BY-NC-4.0",   # NC = non-commercial; verify before any commercial use
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
            "CC-BY-NC-4.0 licence restricts commercial use. "
            "Multilingual base may still benefit cross-lingual transfer, "
            "but Krio tokeniser fertility must be measured first."
        ),
        recommended=False,
        source_url="https://huggingface.co/CohereForAI/aya-23-8B",
    ),

    CandidateModel(
        name="Tiny Aya (Aya-Expanse variants)",
        hf_repo="CohereForAI/aya-expanse-8b",
        parameters="~3.35B–8B (multiple variants)",
        licence="CC-BY-NC-4.0 (verify on model card)",
        multilingual=True,
        african_language_support=(
            "Reported as 70+ languages; 'Earth' and 'Global' variants target African languages. "
            "Krio (kri) coverage is UNVERIFIED — check model card before use."
        ),
        krio_tokenizer_coverage="unknown – needs measurement",
        min_vram_gb=8,
        notes=(
            "Released Feb 2026. Smaller scale (3.35B) makes QLoRA more feasible. "
            "Licence and exact language list must be verified from model card. "
            "Source: TechCrunch (17 Feb 2026), arXiv 2603.11510."
        ),
        recommended=False,
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
            "Verify language list from model card before use. "
            "Sources: AfroBench (arXiv 2311.07978), Lugha-Llama (arXiv 2504.06536)."
        ),
        krio_tokenizer_coverage="unknown – Krio not in reported training languages",
        min_vram_gb=16,
        notes=(
            "Entry requires manual verification of: (1) exact base model (Llama-2 or Llama-3), "
            "(2) parameter count, (3) licence, (4) exact language list. "
            "Do not rely on this entry until model card is read."
        ),
        recommended=False,
        source_url="https://huggingface.co/Jacaranda/AfroLlama_V1",
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
            "QLoRA at 4-bit runs on 16 GB VRAM. Tokeniser fertility must be measured first. "
            "Source: arXiv 2601.06395."
        ),
        recommended=False,
        source_url="https://huggingface.co/meta-llama/Llama-3.1-8B",
    ),

    CandidateModel(
        name="Gemma-3 4B",
        hf_repo="google/gemma-3-4b-pt",
        parameters="4B",
        licence="VERIFY on model card (Gemma Terms of Use)",
        multilingual=False,
        african_language_support=(
            "Used in AfriqueLLM (arXiv 2601.06395, ACL 2026) for African language adaptation. "
            "Krio not among the AfriqueLLM languages; base model has no Krio data."
        ),
        krio_tokenizer_coverage="unknown – needs measurement",
        min_vram_gb=8,
        notes=(
            "Smaller scale; suitable for resource-constrained training. "
            "AfriqueLLM 2026 showed African-language CPT benefit on this architecture. "
            "Verify Gemma Terms of Use for research vs commercial use."
        ),
        recommended=False,
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
            "Krio not among the AfriqueLLM languages; base model has no Krio data."
        ),
        krio_tokenizer_coverage="unknown – needs measurement",
        min_vram_gb=24,
        notes="Larger Gemma-3 variant. Requires 24 GB VRAM for QLoRA. Source: arXiv 2601.06395.",
        recommended=False,
        source_url="https://huggingface.co/google/gemma-3-12b-pt",
    ),

    CandidateModel(
        name="Qwen3-4B-Base",
        hf_repo="Qwen/Qwen3-4B-Base",
        parameters="4B",
        licence="VERIFY on model card",
        multilingual=True,
        african_language_support=(
            "Used in AfriqueLLM (arXiv 2601.06395, ACL 2026). "
            "Krio not among AfriqueLLM languages; Qwen3 multilingual coverage must be verified."
        ),
        krio_tokenizer_coverage="unknown – needs measurement",
        min_vram_gb=8,
        notes="Source: arXiv 2601.06395. Multilingual base; verify Krio tokeniser efficiency.",
        recommended=False,
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
            "Krio not among AfriqueLLM languages; Qwen3 multilingual coverage must be verified."
        ),
        krio_tokenizer_coverage="unknown – needs measurement",
        min_vram_gb=16,
        notes="Source: arXiv 2601.06395. Larger Qwen3 variant.",
        recommended=False,
        source_url="https://huggingface.co/Qwen/Qwen3-8B-Base",
    ),

    CandidateModel(
        name="mT5-base",
        hf_repo="google/mt5-base",
        parameters="580M",
        licence="Apache-2.0",
        multilingual=True,
        african_language_support=(
            "Trained on 101 languages (mC4 corpus). "
            "Krio (kri) not explicitly listed; coverage unknown."
        ),
        krio_tokenizer_coverage="unknown – needs measurement",
        min_vram_gb=4,
        notes=(
            "Encoder-decoder architecture; best suited for translation tasks. "
            "Apache-2.0 is the most permissive licence in this list. "
            "Small scale makes it trainable on modest hardware."
        ),
        recommended=False,
        source_url="https://huggingface.co/google/mt5-base",
    ),
]


def get_recommended() -> CandidateModel | None:
    """
    Return the recommended model if one has been selected after the tokenisation audit.
    Returns None during the research phase (all models are recommended=False).
    """
    for m in CANDIDATE_MODELS:
        if m.recommended:
            return m
    return None


def list_candidates() -> list[CandidateModel]:
    return CANDIDATE_MODELS


# Alias for consistency with test suite
def get_all_candidates() -> list[CandidateModel]:
    """Return all registered candidate models. Alias for list_candidates()."""
    return CANDIDATE_MODELS
