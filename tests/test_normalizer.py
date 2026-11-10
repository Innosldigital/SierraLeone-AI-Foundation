"""Tests for the Krio text normaliser and cross-creole contamination check."""
from sierraleone_ai.preprocessing.krio_normalizer import (
    KrioNormalizer,
    NormalisationConfig,
    detect_non_krio_creole_markers,
    orthography_profile,
)


class TestKrioNormalizer:
    def setup_method(self):
        self.normalizer = KrioNormalizer()

    def test_normalise_whitespace(self):
        # Uses neutral repeated-word English text — not Krio
        result = self.normalizer.normalise("TEST  FIXTURE  TEXT  WHITESPACE.")
        assert result.normalised == "TEST FIXTURE TEXT WHITESPACE."
        assert "whitespace_normalised" in result.changes

    def test_no_change_for_clean_text(self):
        result = self.normalizer.normalise("TEST_FIXTURE_CLEAN_TEXT_001")
        assert result.normalised == "TEST_FIXTURE_CLEAN_TEXT_001"

    def test_truncation(self):
        cfg = NormalisationConfig(max_length=10, on_overlong="truncate")
        normalizer = KrioNormalizer(config=cfg)
        result = normalizer.normalise("A" * 20)
        assert len(result.normalised) == 10
        assert result.truncated is True
        assert result.rejected is False

    def test_rejection_on_overlong(self):
        cfg = NormalisationConfig(max_length=10, on_overlong="reject")
        normalizer = KrioNormalizer(config=cfg)
        result = normalizer.normalise("A" * 20)
        assert result.rejected is True
        assert "rejected_overlong" in result.changes

    def test_original_preserved(self):
        original = "TEST  FIXTURE  ORIGINAL."
        result = self.normalizer.normalise(original)
        assert result.original == original

    def test_flags_nigerian_pidgin_markers(self):
        # "abeg" and "oyibo" are documented Nigerian Pidgin markers — not SL Krio
        result = self.normalizer.normalise("Abeg tell di oyibo man.")
        assert len(result.flagged_non_krio_markers) >= 2
        assert "flagged_non_krio_creole_markers" in result.changes

    def test_clean_neutral_text_not_flagged(self):
        # Neutral English text that contains no Nigerian Pidgin / Ghanaian / Liberian markers
        markers = detect_non_krio_creole_markers(
            "Please go to the market quickly and return."
        )
        assert markers == []

    def test_orthography_profile(self):
        # Text containing standard open vowels ɛ and ɔ
        profile = orthography_profile("ɛnd ɔlso standard")
        assert profile["open_e_count"] == 1
        assert profile["open_o_count"] == 1
        assert profile["total_open_vowels"] == 2
        assert profile["standard_orthography_ratio"] > 0.0
