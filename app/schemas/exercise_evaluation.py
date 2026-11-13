"""Schéma Gemini de la correction d'une réponse à un exercice : aucun score (RG-1)."""

from pydantic import BaseModel, Field


class ExerciseEvaluation(BaseModel):
    feedback: str = Field(description="2-4 encouraging sentences in French: overall assessment, never a grade or a score.")
    correct_points: list[str] = Field(
        default_factory=list, description="What the learner got right, as short items in French."
    )
    missing_points: list[str] = Field(
        default_factory=list, description="What is missing or wrong in the answer, as short items in French."
    )
    expected_approach: str = Field(
        description="The expected approach in French: the key steps of a correct solution, without dumping a grade."
    )
