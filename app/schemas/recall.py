"""Schéma Gemini de l'évaluation d'une reformulation (« explique avec tes mots »)."""

from enum import Enum

from pydantic import BaseModel, Field


class RecallVerdict(str, Enum):
    CORRECT = "correct"
    PARTIEL = "partiel"
    INCORRECT = "incorrect"


class RecallEvaluation(BaseModel):
    verdict: RecallVerdict = Field(
        description="correct: all key ideas present and right; partiel: some; incorrect: none or wrong."
    )
    feedback: str = Field(
        description="2-3 encouraging sentences in French: what is right, what to fix. Never quote the key points list."
    )
    missing_points: list[str] = Field(
        default_factory=list,
        description="Key ideas from the expected list that the answer misses or gets wrong, as short hints in French.",
    )
