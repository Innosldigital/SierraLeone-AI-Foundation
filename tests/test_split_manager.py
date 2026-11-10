"""
Tests for split_manager deterministic_split and speaker_aware_split (SLAI-009 / SLAI-017).
"""
from __future__ import annotations

from pathlib import Path

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

    def test_missing_speaker_id_fails_closed_with_record_ids(self):
        """SLAI-023: Records without speaker_id must fail closed and list offending record IDs."""
        records = [
            {"speaker_id": "spk_01", "id": "rec_001"},
            {"id": "rec_missing_spk"},  # missing speaker_id
        ]
        with pytest.raises(ValueError, match="rec_missing_spk"):
            speaker_aware_split(records, seed=42)

    def test_small_n_three_and_four_speakers_nonempty_splits(self):
        """SLAI-009: n=3 and n=4 speakers must allocate >= 1 speaker to every active split."""
        for n_spk in (3, 4):
            records = self._make_records(n_speakers=n_spk, records_per_speaker=2)
            splits = speaker_aware_split(records, seed=42)
            for name, split_recs in splits.items():
                assert len(split_recs) >= 2, f"n={n_spk}: split '{name}' was empty"

    def test_fewer_speakers_than_splits_raises_value_error(self):
        """SLAI-009: 2 speakers across 3 splits must raise ValueError."""
        records = self._make_records(n_speakers=2, records_per_speaker=2)
        with pytest.raises(ValueError, match="need at least"):
            speaker_aware_split(records, seed=42)

    def test_speaker_manifest_persistence_and_reuse_on_new_speaker(self, tmp_path: Path):
        """SLAI-023: Persisted speaker manifest preserves prior speaker splits when a new speaker arrives."""
        manifest_path = tmp_path / "speaker_splits_v1.json"
        initial_records = self._make_records(n_speakers=6, records_per_speaker=2)
        splits_v1 = speaker_aware_split(
            initial_records,
            seed=42,
            manifest_path=manifest_path,
            persist_manifest=True,
        )
        assert manifest_path.exists()

        # Map each original speaker to its v1 split
        v1_assignment = {
            r["speaker_id"]: split_name
            for split_name, recs in splits_v1.items()
            for r in recs
        }

        # Add a new speaker spk_99
        expanded_records = initial_records + [
            {"speaker_id": "spk_99", "id": 9901},
            {"speaker_id": "spk_99", "id": 9902},
        ]
        splits_v2 = speaker_aware_split(
            expanded_records,
            seed=42,
            manifest_path=manifest_path,
            persist_manifest=True,
        )
        v2_assignment = {
            r["speaker_id"]: split_name
            for split_name, recs in splits_v2.items()
            for r in recs
        }

        for spk, original_split in v1_assignment.items():
            assert v2_assignment[spk] == original_split
        assert "spk_99" in v2_assignment

    def test_held_out_speakers_permanently_excluded(self, tmp_path: Path):
        """SLAI-023: Speakers marked held_out in manifest must never enter train/validation/test."""
        import json

        manifest_path = tmp_path / "speaker_splits_v1.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "manifest_version": 1,
                    "seed": 42,
                    "ratios": {"train": 0.8, "validation": 0.1, "test": 0.1},
                    "speaker_to_split": {"spk_held_out": "held_out"},
                    "held_out_speakers": ["spk_held_out"],
                }
            ),
            encoding="utf-8",
        )
        records = self._make_records(n_speakers=4, records_per_speaker=2) + [
            {"speaker_id": "spk_held_out", "id": 888}
        ]
        splits = speaker_aware_split(records, seed=42, manifest_path=manifest_path)
        assigned_speakers = {
            r["speaker_id"] for recs in splits.values() for r in recs
        }
        assert "spk_held_out" not in assigned_speakers


