"""Tests for evaluation metrics and native-speaker evaluation forms."""

import pytest

from sierraleone_ai.evaluation.metrics import (
    bleu_score,
    character_error_rate,
    chrf_score,
    word_error_rate,
)
from sierraleone_ai.evaluation.native_speaker import KrioFluencyEvaluation, LikertScore


class TestBLEU:
    def test_perfect_match(self):
        ref = [["this", "is", "a", "test"]]
        hyp = ["this", "is", "a", "test"]
        assert bleu_score(ref, hyp) == pytest.approx(1.0, abs=1e-6)

    def test_empty_hypothesis(self):
        ref = [["this", "is", "a", "test"]]
        assert bleu_score(ref, []) == 0.0

    def test_short_hypothesis_scores_nonzero(self):
        """Short hypotheses (<4 tokens) must not return 0.0 on a full match."""
        ref = [["hello"]]
        hyp = ["hello"]
        score = bleu_score(ref, hyp)
        assert score > 0.0

    def test_empty_vs_empty_returns_zero(self):
        """Both empty → hypothesis is empty → return 0.0 by convention."""
        assert bleu_score([[]], []) == 0.0

    def test_two_token_exact_match_is_one(self):
        """SLAI-008: bleu_score([["a", "b"]], ["a", "b"]) == 1.0."""
        assert bleu_score([["a", "b"]], ["a", "b"]) == pytest.approx(1.0, abs=1e-6)

    def test_empty_references_raises_value_error(self):
        """SLAI-008: Empty references list raises ValueError."""
        with pytest.raises(ValueError, match="references must be a non-empty"):
            bleu_score([], ["a", "b"])


class TestChrF:
    def test_chrf_perfect_match(self):
        assert chrf_score(["hello world"], "hello world") == pytest.approx(1.0, abs=1e-6)

    def test_chrf_partial_match(self):
        score = chrf_score(["hello world"], "hello earth")
        assert 0.0 < score < 1.0

    def test_chrf_empty_references_raises(self):
        with pytest.raises(ValueError, match="references must not be empty"):
            chrf_score([], "hello")


class TestCER:
    def test_perfect_match(self):
        assert character_error_rate("hello", "hello") == pytest.approx(0.0)

    def test_full_error(self):
        cer = character_error_rate("abc", "xyz")
        assert cer > 0.0

    def test_both_empty_returns_zero(self):
        assert character_error_rate("", "") == pytest.approx(0.0)

    def test_empty_reference_nonempty_hyp_returns_inf(self):
        assert character_error_rate("", "abc") == float("inf")


class TestWER:
    def test_perfect_match(self):
        assert word_error_rate("hello world", "hello world") == pytest.approx(0.0)

    def test_one_substitution(self):
        wer = word_error_rate("hello world", "hello earth")
        assert wer == pytest.approx(0.5)

    def test_both_empty_returns_zero(self):
        assert word_error_rate("", "") == pytest.approx(0.0)

    def test_empty_reference_nonempty_hyp_returns_inf(self):
        assert word_error_rate("", "some words") == float("inf")


class TestNativeSpeakerEvaluation:
    def test_mean_score_calculation(self):
        # Prompt and output fields are fixture strings, not real Krio
        form = KrioFluencyEvaluation(
            evaluation_id="eval_001",
            evaluator_id="native_reviewer_01",
            evaluator_native_krio=True,
            model_id="cafp-krio-ai",
            model_version="0.1.0-alpha",
            prompt="FIXTURE_PROMPT_001",
            model_output="FIXTURE_OUTPUT_001",
            grammatical_correctness=LikertScore.EXCELLENT,
            vocabulary_appropriateness=LikertScore.GOOD,
            naturalness=LikertScore.EXCELLENT,
            cultural_appropriateness=LikertScore.EXCELLENT,
            task_completion=LikertScore.GOOD,
        )
        assert form.mean_score == pytest.approx(4.6)

    def test_low_score_requires_notes(self):
        """A score of VERY_POOR without notes must raise a validation error."""
        with pytest.raises(Exception, match="evaluator_notes"):
            KrioFluencyEvaluation(
                evaluation_id="eval_002",
                evaluator_id="native_reviewer_01",
                evaluator_native_krio=True,
                model_id="cafp-krio-ai",
                model_version="0.1.0-alpha",
                prompt="FIXTURE_PROMPT_002",
                model_output="FIXTURE_OUTPUT_002",
                grammatical_correctness=LikertScore.VERY_POOR,
                # evaluator_notes intentionally omitted → must fail
            )

    def test_low_score_with_notes_accepted(self):
        """A score of POOR with notes must be accepted."""
        form = KrioFluencyEvaluation(
            evaluation_id="eval_003",
            evaluator_id="native_reviewer_01",
            evaluator_native_krio=True,
            model_id="cafp-krio-ai",
            model_version="0.1.0-alpha",
            prompt="FIXTURE_PROMPT_003",
            model_output="FIXTURE_OUTPUT_003",
            grammatical_correctness=LikertScore.POOR,
            evaluator_notes="FIXTURE_NOTE: grammar issue detected in output",
        )
        assert form.grammatical_correctness == LikertScore.POOR


class TestReporting:
    def test_compute_sacrebleu(self):
        """PA-018: Test official sacrebleu reporting and signatures."""
        from sierraleone_ai.evaluation.reporting import compute_sacrebleu

        try:
            import sacrebleu  # noqa: F401
        except ImportError:
            with pytest.raises(RuntimeError, match="requires `sacrebleu`"):
                compute_sacrebleu(["test"], [["test"]])
            return

        hyp = ["this is a test"]
        ref = [["this is a test"]]
        res = compute_sacrebleu(hyp, ref)
        assert "bleu_score" in res
        assert "bleu_signature" in res
        assert "chrf_score" in res
        assert "chrf_signature" in res
        assert res["bleu_score"] == pytest.approx(100.0, abs=1e-2)
        assert res["chrf_score"] == pytest.approx(100.0, abs=1e-2)
        assert "BLEU" in res["bleu_signature"] or "nrefs" in res["bleu_signature"]
