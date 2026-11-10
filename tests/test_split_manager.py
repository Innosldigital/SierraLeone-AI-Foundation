"""
Tests for split_manager deterministic_split and speaker_aware_split (SLAI-009 / SLAI-017).
"""
from __future__ import annotations

import pytest

from sierraleone_ai.preprocessing.split_manager import (
    deterministic_split,
    speaker_aware_split,
)


class TestDeterministicSplit:
    def test_total_items_preserved(self):
        items = list(range(100))
        splits = deterministic_split(items)
        total = sum(len(v) for v in splits.values())
        assert total == 100

    def test_default_ratio_names(self):
        splits = deterministic_split(list(range(10)))
        assert set(splits.keys()) == {"train", "validation", "test"}

    def test_reproducibility(self):
        items = list(range(50))
        s1 = deterministic_split(items, seed=99)
        s2 = deterministic_split(items, seed=99)
        assert s1 == s2

    def test_different_seeds_differ(self):
        items = list(range(50))
        s1 = deterministic_split(items, seed=1)
        s2 = deterministic_split(items, seed=2)
        # With high probability the train splits differ
        assert s1["train"] != s2["train"]

    def test_invalid_ratios_raise_value_error(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            deterministic_split(list(range(10)), ratios={"train": 0.9, "test": 0.5})

    def test_negative_ratio_raises_value_error(self):
        with pytest.raises(ValueError):
            deterministic_split(list(range(10)), ratios={"train": -0.1, "test": 1.1})

    def test_all_splits_nonempty_for_sufficient_items(self):
        items = list(range(30))
        splits = deterministic_split(items)
        for name, split_items in splits.items():
            assert len(split_items) >= 1, f"Split '{name}' is empty"

    def test_custom_ratios(self):
        items = list(range(100))
        ratios = {"a": 0.6, "b": 0.4}
        splits = deterministic_split(items, ratios=ratios)
        assert len(splits["a"]) + len(splits["b"]) == 100
        # Largest-remainder: exact split at 60/40
        assert abs(len(splits["a"]) - 60) <= 1

    def test_largest_remainder_no_data_loss(self):
        """Total must always equal len(items) regardless of rounding."""
        for n in [7, 11, 13, 100, 1001]:
            items = list(range(n))
            splits = deterministic_split(items)
            assert sum(len(v) for v in splits.values()) == n


class TestSpeakerAwareSplit:
    def _make_records(self, n_speakers: int = 9, records_per_speaker: int = 3) -> list[dict]:
        return [
            {"speaker_id": f"spk_{s:02d}", "id": s * records_per_speaker + r}
            for s in range(n_speakers)
            for r in range(records_per_speaker)
        ]

    def test_no_speaker_leakage(self):
        records = self._make_records()
        splits = speaker_aware_split(records, seed=42)
        train_spk = {r["speaker_id"] for r in splits["train"]}
        val_spk = {r["speaker_id"] for r in splits["validation"]}
        test_spk = {r["speaker_id"] for r in splits["test"]}
        assert train_spk.isdisjoint(val_spk), "Speaker leakage: train ∩ validation"
        assert train_spk.isdisjoint(test_spk), "Speaker leakage: train ∩ test"
        assert val_spk.isdisjoint(test_spk), "Speaker leakage: validation ∩ test"

    def test_all_records_assigned(self):
        records = self._make_records()
        splits = speaker_aware_split(records, seed=42)
        total = sum(len(v) for v in splits.values())
        assert total == len(records)

    def test_reproducible(self):
        records = self._make_records()
        s1 = speaker_aware_split(records, seed=7)
        s2 = speaker_aware_split(records, seed=7)
        assert s1 == s2

    def test_anonymous_records_handled(self):
        """Records without speaker_id should be distributed across splits without error."""
        records = [{"id": i} for i in range(30)]  # no speaker_id
        splits = speaker_aware_split(records, seed=42, allow_missing_speaker_id=True)
        total = sum(len(v) for v in splits.values())
        assert total == len(records)

