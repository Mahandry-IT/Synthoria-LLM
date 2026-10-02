"""Schéma Gemini de l'évaluation d'une réponse au défi d'une section (posé avant l'explication)."""

from enum import Enum

from pydantic import BaseModel, Field


class ChallengeVerdict(str, Enum):
    ON_TRACK = "on_track"
    PARTIAL = "partial"
    OFF_TRACK = "off_track"


class ChallengeEvaluation(BaseModel):
    verdict: ChallengeVerdict = Field(
        description=(
            "on_track: the reasoning heads toward the expected ideas; partial: some right intuitions, "
            "something important missing or wrong; off_track: wrong direction or no real attempt."
        )
    )
    feedback: str = Field(
        description=(
            "1-2 encouraging sentences in French on the learner's reasoning. Never give the answer, "
            "never explain the concept, never quote the expected ideas."
        )
    )
    hint: str = Field(
        description=(
            "One short guiding question or clue in French that points toward the explanation that "
            "follows, without revealing it."
        )
    )
