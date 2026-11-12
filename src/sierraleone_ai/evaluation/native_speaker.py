"""
Native speaker evaluation framework.

This module defines the structured evaluation forms and rating protocols
for human evaluation of Krio language model outputs.

Native speaker evaluation is mandatory before any public model claims.
Automated metrics alone are INSUFFICIENT to validate Krio language quality.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import IntEnum

from pydantic import AliasChoices, BaseModel, Field, model_validator


class LikertScore(IntEnum):
    """Five-point Likert scale used across all human evaluation dimensions."""

    VERY_POOR = 1
    POOR = 2
    ACCEPTABLE = 3
    GOOD = 4
    EXCELLENT = 5


class KrioFluencyEvaluation(BaseModel):
    """
    Native speaker evaluation form for a single model output.

    Instructions for evaluators:
    - Read each model output carefully.
    - Score each dimension independently.
    - Provide written justification for scores of 1 or 2.
    - Flag any culturally inappropriate content.
    - Do not rate outputs you are unsure about – leave them as None.
    """

    # Administrative
    evaluation_id: str
    evaluator_id: str = Field(..., description="Anonymised evaluator identifier")
    evaluator_native_krio: bool = Field(
        ...,
        validation_alias=AliasChoices("evaluator_native_krio", "evaluator_native_speaker"),
        description="Is the evaluator a native Sierra Leonean Krio speaker?",
    )
    evaluation_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Model output under evaluation
    model_id: str
    model_version: str
    prompt: str
    model_output: str
    reference_output: str | None = Field(None, description="Human reference if available")

    # Evaluation dimensions
    grammatical_correctness: LikertScore | None = Field(
        None,
        description="Does the text follow Krio grammar rules? (1=Very Poor, 5=Excellent)",
    )
    vocabulary_appropriateness: LikertScore | None = Field(
        None,
        description="Are the words natural Sierra Leonean Krio? Not Nigerian Pidgin or other creoles.",
    )
    naturalness: LikertScore | None = Field(
        None,
        description="Does this sound like how a real Krio speaker would say it?",
    )
    cultural_appropriateness: LikertScore | None = Field(
        None,
        description="Is the content culturally appropriate for Sierra Leone?",
    )
    task_completion: LikertScore | None = Field(
        None,
        description="Did the model complete the requested task correctly?",
    )

    # Flags
    contains_non_krio_creole: bool = Field(
        False,
        description="True if the output contains Nigerian Pidgin, Liberian English or other non-SL creole",
    )
    contains_harmful_content: bool = False
    contains_factual_errors: bool = False

    # Free text
    evaluator_notes: str | None = None
    suggested_correction: str | None = None

    @model_validator(mode="after")
    def low_score_requires_notes(self) -> KrioFluencyEvaluation:
        """Scores of VERY_POOR or POOR must be accompanied by written notes."""
        scored = [
            s
            for s in [
                self.grammatical_correctness,
                self.vocabulary_appropriateness,
                self.naturalness,
                self.cultural_appropriateness,
                self.task_completion,
            ]
            if s is not None
        ]
        if any(s <= LikertScore.POOR for s in scored) and not self.evaluator_notes:
            raise ValueError(
                "evaluator_notes is required when any dimension score is 1 (Very Poor) "
                "or 2 (Poor). Please provide written justification."
            )
        return self

    @property
    def mean_score(self) -> float | None:
        """Mean score across all rated dimensions."""
        scores = [
            s
            for s in [
                self.grammatical_correctness,
                self.vocabulary_appropriateness,
                self.naturalness,
                self.cultural_appropriateness,
                self.task_completion,
            ]
            if s is not None
        ]
        return sum(scores) / len(scores) if scores else None


class TranslationEvaluation(BaseModel):
    """Human evaluation form for translation quality."""

    evaluation_id: str
    evaluator_id: str
    evaluator_native_krio: bool
    evaluation_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    source_text: str
    source_language: str
    model_translation: str
    reference_translation: str | None = None
    target_language: str

    adequacy: LikertScore | None = Field(
        None, description="Does the translation convey the full meaning of the source?"
    )
    fluency: LikertScore | None = Field(
        None, description="Is the translation natural and grammatical Krio?"
    )
    terminology: LikertScore | None = Field(
        None, description="Are domain-specific terms correctly handled?"
    )

    evaluator_notes: str | None = None
    suggested_translation: str | None = None
