"""
Tests for the model registry (SLAI-005 / SLAI-017).

Verifies that:
- get_recommended() returns None (no model is approved yet).
- Every candidate has hf_repo and source_url fields populated.
- Aya-23 African language support is not falsely claimed.
"""
from __future__ import annotations

import pytest

from sierraleone_ai.models.registry import get_all_candidates, get_recommended


class TestRegistry:
    def test_get_recommended_returns_none(self):
        """No model should be recommended until all 5 approval criteria are met."""
        assert get_recommended() is None

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

    def test_aya23_does_not_claim_sub_saharan_african_support(self):
        """
        Aya-23 covers 23 languages – none in sub-Saharan Africa.
        This test guards against the factual error that was present before SLAI-005.
        """
        aya = next(
            (c for c in get_all_candidates() if "aya" in c.name.lower() and "23" in c.name),
            None,
        )
        if aya is None:
            pytest.skip("Aya-23 not in registry – skip")
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

