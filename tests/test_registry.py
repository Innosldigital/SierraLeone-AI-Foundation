"""
Tests for the model registry (SLAI-005 / SLAI-017 / PA-005).

Verifies that:
- get_recommended() returns None (no model is approved yet).
- Every candidate has hf_repo and source_url fields populated.
- Aya-23 African language support is not falsely claimed.
- Documented Krio support requires evidence_url.
- Tiny Aya and DONDO checkpoints are registered with truthful descriptions.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from sierraleone_ai.models.registry import (
    CandidateModel,
    get_all_candidates,
    get_recommended,
    list_candidates,
)


class TestRegistry:
    def test_get_recommended_returns_none(self):
        """No model should be recommended until all 5 approval criteria are met."""
        assert get_recommended() is None

    def test_get_recommended_reads_decision_file(self, tmp_path: Path):
        """SLAI-032: get_recommended reads models/selected_base_model.json when present."""
        decision = tmp_path / "selected_base_model.json"
        decision.write_text(
            json.dumps(
                {
                    "hf_repo": "Qwen/Qwen3-4B",
                    "verified_on": "2025-01-15",
                    "tokenizer_audit_completed": True,
                    "native_benchmark_completed": True,
                }
            ),
            encoding="utf-8",
        )
        selected = get_recommended(decision_file=decision)
        assert selected is not None
        assert selected.hf_repo == "Qwen/Qwen3-4B"

    def test_task_separation_and_2025_candidates(self):
        """SLAI-032 / PA-005: Candidates are separated by task and include verified models."""
        all_repos = {c.hf_repo for c in get_all_candidates()}
        for expected_repo in (
            "Qwen/Qwen3-4B",
            "google/gemma-3-1b-pt",
            "google/gemma-3-4b-pt",
            "google/gemma-3-12b-pt",
            "meta-llama/Llama-3.2-1B",
            "meta-llama/Llama-3.2-3B",
            "facebook/nllb-200-distilled-600M",
            "openai/whisper-small",
            "facebook/mms-tts",
            "CohereLabs/tiny-aya-base",
            "CohereLabs/tiny-aya-global",
            "CohereLabs/tiny-aya-earth",
            "KhayaAI/w2v-bert-men",
            "KhayaAI/w2v-bert-tem",
        ):
            assert expected_repo in all_repos, f"Missing {expected_repo} in registry"

        assert len(list_candidates("chat_llm")) >= 6
        assert len(list_candidates("translation_seq2seq")) >= 1
        assert len(list_candidates("asr")) >= 3
        assert len(list_candidates("tts")) >= 1

    def test_every_candidate_has_hf_repo(self):
        for candidate in get_all_candidates():
            assert candidate.hf_repo, (
                f"Candidate '{candidate.name}' has no hf_repo set"
            )

    def test_every_candidate_has_source_url(self):
        for candidate in get_all_candidates():
            assert candidate.source_url, (
                f"Candidate '{candidate.name}' has no source_url set"
            )

    def test_krio_support_documented_requires_evidence_url(self):
        """PA-005: Candidate with krio_support='documented' must have evidence_url."""
        for candidate in get_all_candidates():
            if candidate.krio_support == "documented":
                assert candidate.evidence_url, (
                    f"Candidate '{candidate.name}' has krio_support='documented' but lacks evidence_url"
                )

        with pytest.raises(ValidationError, match="missing evidence_url"):
            CandidateModel(
                name="Test Model",
                hf_repo="test/model",
                parameters="1B",
                licence="MIT",
                african_language_support="test",
                krio_support="documented",
                min_vram_gb=4,
                source_url="https://example.org",
                evidence_url=None,
            )

    def test_verified_model_requires_verified_on(self):
        """PA-005: verified=True requires verified_on ISO date string."""
        with pytest.raises(ValidationError, match="missing verified_on date"):
            CandidateModel(
                name="Test Model",
                hf_repo="test/model",
                parameters="1B",
                licence="MIT",
                african_language_support="test",
                min_vram_gb=4,
                source_url="https://example.org",
                verified=True,
                verified_on=None,
            )

    def test_nllb_notes_krio_not_in_official_card(self):
        """PA-005: NLLB entry must state that Krio is NOT on the official model card."""
        nllb = next(
            (c for c in get_all_candidates() if "nllb" in c.name.lower()),
            None,
        )
        assert nllb is not None
        assert "not in the nllb-200 language list" in nllb.african_language_support.lower()

    def test_aya23_does_not_claim_sub_saharan_african_support(self):
        """
        Aya-23 covers 23 languages – none in sub-Saharan Africa.
        This test guards against the factual error that was present before SLAI-005.
        """
        aya = next(
            (c for c in get_all_candidates() if "aya" in c.name.lower() and "23" in c.name),
            None,
        )
        assert aya is not None, "Aya-23 must be registered"
        # Must explicitly note lack of sub-Saharan African support / lack of Krio
        support_lower = (aya.african_language_support or "").lower()
        assert "no sub-saharan" in support_lower or "none" in support_lower, (
            "Aya-23 registry entry must state no sub-Saharan African language support"
        )
        assert "krio is not covered" in support_lower or "not covered" in support_lower, (
            "Aya-23 registry entry must note that Krio is not covered"
        )

    def test_at_least_five_candidates(self):
        candidates = get_all_candidates()
        assert len(candidates) >= 5, f"Expected ≥5 candidates, got {len(candidates)}"

