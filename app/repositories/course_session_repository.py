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
        gemini_response=response.model_dump(exclude={"session_id", "podcast_job_id"}),
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


async def update_section(
    session: AsyncSession,
    session_id: uuid.UUID,
    section_id: str,
    updated_section: dict[str, Any],
) -> CourseSession | None:
    """Remplace une section (par `id`) dans `gemini_response.sections`.

    None si la session ou la section est introuvable. Réassigne tout le dict `gemini_response`
    (plutôt qu'une mutation en place) : `Mapped[dict]` sans `MutableDict` ne détecterait pas une
    mutation en place et ne la persisterait pas.
    """
    row = await get_by_id(session, session_id)
    if row is None:
        return None
    sections = row.gemini_response.get("sections") or []
    index = next((i for i, s in enumerate(sections) if str(s.get("id")) == section_id), None)
    if index is None:
        return None
    row.gemini_response = {
        **row.gemini_response,
        "sections": [*sections[:index], updated_section, *sections[index + 1 :]],
    }
    await session.commit()
    await session.refresh(row)
    return row
