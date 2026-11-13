"""Schéma Gemini du lot d'exercices corrigés d'un cours (UN seul appel pour tout le lot, RG-9)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.schemas.course_generation import ContentBlock

HINTS_COUNT = 3
EXERCISE_MAX_HARD_CAP = 10  # borne absolue du JSON schema ; le plafond du mode s'applique ensuite


class ExerciseDraft(BaseModel):
    section_id: str | None = Field(
        description=(
            "`id` of the course section this exercise practises, copied VERBATIM from the <sections> list; "
            "null only if it spans the whole course."
        )
    )
    title: str = Field(description="Short title of the exercise (no solution in it).")
    covers: str = Field(
        description="One sentence in French telling what skill or notion the exercise practises, WITHOUT revealing the statement or the solution."
    )
    statement: list[ContentBlock] = Field(
        description=(
            "The exercise statement as ordered content blocks (text, table, formula, chart, code...). Concrete, "
            "with real data. Never contains the solution or the answer. No IMAGE blocks."
        )
    )
    hints: list[str] = Field(
        description="Exactly 3 hints in French, from the most general to the most precise. The last hint never gives the final answer."
    )
    duration_minutes: int = Field(description="Realistic time to solve it, in minutes (5 to 60).")
    expected_solution: str = Field(
        description=(
            "The full correct solution in French, step by step, with the final result. Confidential: used to grade "
            "the learner's answer, never shown before they answered."
        )
    )


class ExerciseBatch(BaseModel):
    exercises: list[ExerciseDraft] = Field(
        description="All the exercises of the course, in one list. Empty if no section really needs one (RG-6)."
    )

    @classmethod
    def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Schéma envoyé à Gemini : tous les champs d'un exercice sont obligatoires (sinon Flash-Lite en
        omet), 3 indices exactement, au moins un bloc d'énoncé. La validation Pydantic reste lâche : le
        service écarte les exercices invalides un à un plutôt que de tout rejeter."""
        schema = super().model_json_schema(*args, **kwargs)
        draft = schema.get("$defs", {}).get("ExerciseDraft", {})
        draft["required"] = list(draft.get("properties", {}))
        props = draft.get("properties", {})
        if "hints" in props:
            props["hints"].update(minItems=HINTS_COUNT, maxItems=HINTS_COUNT)
        if "statement" in props:
            props["statement"]["minItems"] = 1
        return schema


def exercise_batch_json_schema(max_exercises: int) -> dict[str, Any]:
    """JSON schema du lot, plafonné à `max_exercises` exercices (plafond du mode du cours)."""
    schema = ExerciseBatch.model_json_schema()
    schema["properties"]["exercises"]["maxItems"] = max(0, min(max_exercises, EXERCISE_MAX_HARD_CAP))
    return schema
