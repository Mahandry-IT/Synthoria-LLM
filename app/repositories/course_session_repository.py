import uuid
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import CourseGenerationResponse
from app.db.models import CourseSession


async def save(
    session: AsyncSession,
    *,
    question: str,
    filenames: list[str],
    mode: str,
    response: CourseGenerationResponse,
) -> CourseSession:
    """Persiste une session de génération de cours."""
    course_session = CourseSession(
        question=question,
        filenames=filenames,
        mode=mode,
        gemini_response=response.model_dump(),
    )
    session.add(course_session)
    await session.commit()
    await session.refresh(course_session)
    return course_session


async def list_paginated(
    session: AsyncSession,
    *,
    page: int,
    limit: int,
) -> tuple[list[CourseSession], int]:
    """Liste paginée des sessions (plus récentes en premier)."""
    offset = (page - 1) * limit

    # Total
    count_result = await session.execute(select(func.count(CourseSession.id)))
    total = count_result.scalar_one()

    # Data
    stmt = (
        select(CourseSession)
        .order_by(CourseSession.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    result = await session.execute(stmt)
    rows = list(result.scalars().all())

    return rows, total


async def get_by_id(
    session: AsyncSession,
    session_id: uuid.UUID,
) -> CourseSession | None:
    """Récupère une session par son UUID."""
    result = await session.execute(
        select(CourseSession).where(CourseSession.id == session_id)
    )
    return result.scalar_one_or_none()
