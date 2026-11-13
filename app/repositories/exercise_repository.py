"""Accès à la table `course_exercises`."""

import uuid
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import CourseExercise


async def list_by_session(session: AsyncSession, session_id: uuid.UUID) -> list[CourseExercise]:
    result = await session.execute(
        select(CourseExercise).where(CourseExercise.session_id == session_id).order_by(CourseExercise.position)
    )
    return list(result.scalars().all())


async def get_for_session(
    session: AsyncSession, session_id: uuid.UUID, exercise_id: uuid.UUID
) -> CourseExercise | None:
    """Exercice `exercise_id` seulement s'il appartient à `session_id` (jamais l'id seul : anti-IDOR)."""
    result = await session.execute(
        select(CourseExercise).where(CourseExercise.id == exercise_id, CourseExercise.session_id == session_id)
    )
    return result.scalar_one_or_none()


async def replace_all(session: AsyncSession, session_id: uuid.UUID, rows: list[dict[str, Any]]) -> list[CourseExercise]:
    """Remplace les exercices du cours dans la transaction courante, SANS valider : l'appelant valide
    avec le passage du job à `done` (`exercise_job_repository.mark_done`, appelé avant), d'un seul commit."""
    await session.execute(delete(CourseExercise).where(CourseExercise.session_id == session_id))
    exercises = [CourseExercise(session_id=session_id, position=i, **row) for i, row in enumerate(rows, start=1)]
    session.add_all(exercises)
    return exercises
