import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import CoursePlan


async def save(
    session: AsyncSession,
    *,
    question: str,
    mode: str,
    filenames: list[str],
    top_k: int,
    full_document: bool,
    retrieval_context: dict[str, Any],
    plan: dict[str, Any],
    expires_at: datetime,
) -> CoursePlan:
    """Persiste un plan de cours proposé (statut `pending`)."""
    course_plan = CoursePlan(
        question=question,
        mode=mode,
        filenames=filenames,
        top_k=top_k,
        full_document=full_document,
        retrieval_context=retrieval_context,
        plan=plan,
        expires_at=expires_at,
    )
    session.add(course_plan)
    await session.commit()
    await session.refresh(course_plan)
    return course_plan


async def get_by_id(session: AsyncSession, plan_id: uuid.UUID) -> CoursePlan | None:
    """Récupère un plan par son UUID (None si absent)."""
    result = await session.execute(select(CoursePlan).where(CoursePlan.id == plan_id))
    return result.scalar_one_or_none()


async def mark_generated(session: AsyncSession, plan_id: uuid.UUID) -> None:
    """Marque un plan comme ayant servi à générer un cours (no-op si absent)."""
    course_plan = await get_by_id(session, plan_id)
    if course_plan is None:
        return
    course_plan.status = "generated"
    await session.commit()
