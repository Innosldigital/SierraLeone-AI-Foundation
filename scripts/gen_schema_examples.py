"""
Generate schema examples for documentation/DATASET_FORMAT.md (PA-017).

Uses Pydantic schema v2 models with neutral TEST_FIXTURE strings.
"""

from __future__ import annotations

import json

from sierraleone_ai.datasets.schema import (
    AudioRecord,
    BenchmarkItem,
    ConsentScope,
    ConversationRecord,
    ConversationTurn,
    DatasetMetadata,
    KrioTextRecord,
    Language,
    ReviewerSignoff,
    TranslationRecord,
    ValidationStatus,
)


def make_metadata(consent_scope: list[ConsentScope]) -> DatasetMetadata:
    return DatasetMetadata(
        schema_version=2,
        record_id="01234567-89ab-4cde-8f01-23456789abcd",
        language=Language.KRIO,
        source="test_fixture_approved_source",
        collection_date="2026-01-15",
        licence="CC-BY-4.0",
        licence_id="CC-BY-4.0",
        validation_status=ValidationStatus.VALIDATED,
        dataset_version="0.2.0",
        split="train",
        contributor_id="contrib_001",
        contributor_consent=True,
        consent_scope=consent_scope,
        access_class="public",
        training_permitted=True,
        redistribution_permitted=True,
        reviews=[
            ReviewerSignoff(
                reviewer_id="rev_sl_001",
                is_native_speaker_of_record_language=True,
                reviewed_on="2026-01-16",
                verdict="approve",
            ),
            ReviewerSignoff(
                reviewer_id="rev_sl_002",
                is_native_speaker_of_record_language=True,
                reviewed_on="2026-01-17",
                verdict="approve",
            ),
        ],
    )


def gen_examples() -> dict[str, str]:
    text_rec = KrioTextRecord(
        metadata=make_metadata(["text_training"]),
        text="TEST_FIXTURE_KRIO_TEXT_001",
        script="Latin",
        domain="agriculture",
    )

    trans_rec = TranslationRecord(
        metadata=make_metadata(["text_training"]),
        source_language="english",
        target_language="krio",
        source_text="TEST_FIXTURE_ENGLISH_SOURCE_001",
        target_text="TEST_FIXTURE_KRIO_TARGET_001",
        translation_method="human",
        back_translation_verified=True,
    )

    conv_rec = ConversationRecord(
        metadata=make_metadata(["text_training"]),
        conversation_id="00000000-0000-4000-8000-000000000002",
        turns=[
            ConversationTurn(
                speaker="user", turn_index=0, language="krio", text="TEST_FIXTURE_USER_PROMPT_001"
            ),
            ConversationTurn(
                speaker="assistant",
                turn_index=1,
                language="krio",
                text="TEST_FIXTURE_ASSISTANT_RESPONSE_001",
            ),
        ],
        domain="general",
    )

    audio_rec = AudioRecord(
        metadata=make_metadata(["speech_recognition", "voice_synthesis"]),
        audio_file_path="audio/spk_01/recording_001.wav",
        duration_seconds=3.5,
        speaker_id="spk_01",
        transcript="TEST_FIXTURE_TRANSCRIPT_001",
        sample_rate_hz=24000,
        channels=1,
        snr_db=32.0,
        recording_environment="studio",
    )

    bench_item = BenchmarkItem(
        metadata=DatasetMetadata(
            schema_version=2,
            record_id="00000000-0000-4000-8000-000000000003",
            language=Language.KRIO,
            source="test_fixture_benchmark_collection",
            collection_date="2026-01-20",
            licence="CC-BY-4.0",
            licence_id="CC-BY-4.0",
            validation_status=ValidationStatus.VALIDATED,
            dataset_version="0.2.0",
            split="held_out",
            contributor_id="contrib_bench_01",
            contributor_consent=True,
            consent_scope=["evaluation"],
            access_class="public",
            training_permitted=False,
            redistribution_permitted=True,
            reviews=[
                ReviewerSignoff(
                    reviewer_id="rev_sl_001",
                    is_native_speaker_of_record_language=True,
                    reviewed_on="2026-01-21",
                    verdict="approve",
                ),
                ReviewerSignoff(
                    reviewer_id="rev_sl_002",
                    is_native_speaker_of_record_language=True,
                    reviewed_on="2026-01-22",
                    verdict="approve",
                ),
            ],
        ),
        item_id="bench_krio_001",
        evaluation_category="translation",
        prompt="TEST_FIXTURE_BENCHMARK_PROMPT_001",
        references=["TEST_FIXTURE_BENCHMARK_REF_001"],
        reference_notes="Standard Freetown dialect variant",
        review_count=2,
        benchmark_version="1.0.0",
    )

    return {
        "text": json.dumps(text_rec.model_dump(mode="json"), indent=2),
        "translation": json.dumps(trans_rec.model_dump(mode="json"), indent=2),
        "conversation": json.dumps(conv_rec.model_dump(mode="json"), indent=2),
        "audio": json.dumps(audio_rec.model_dump(mode="json"), indent=2),
        "benchmark": json.dumps(bench_item.model_dump(mode="json"), indent=2),
    }


if __name__ == "__main__":
    ex = gen_examples()
    for name, s in ex.items():
        print(f"=== {name} ===")
        print(s)
