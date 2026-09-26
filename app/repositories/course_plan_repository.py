import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.text_sanitize import sanitize_json
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
    """Persiste un plan de cours proposé (statut `pending`).

    `retrieval_context`/`plan` mélangent du contenu extrait de PDF et du texte généré par le
    LLM : l'un ou l'autre peut contenir un caractère de contrôle (ex. NUL) que Postgres refuse
    dans une colonne `jsonb` (`UntranslatableCharacterError`) — voir `sanitize_json`.
    """
    course_plan = CoursePlan(
        question=question,
        mode=mode,
        filenames=filenames,
        top_k=top_k,
        full_document=full_document,
        retrieval_context=sanitize_json(retrieval_context),
        plan=sanitize_json(plan),
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


async def delete(session: AsyncSession, plan_id: uuid.UUID) -> bool:
    """Supprime un plan proposé. Retourne False si déjà absent (y compris expiré : la suppression
    n'a pas la même contrainte de validité que la lecture)."""
    row = await get_by_id(session, plan_id)
    if row is None:
        return False
    await session.delete(row)
    await session.commit()
    return True


async def mark_generated(session: AsyncSession, plan_id: uuid.UUID) -> None:
    """Marque un plan comme ayant servi à générer un cours (no-op si absent)."""
    course_plan = await get_by_id(session, plan_id)
    if course_plan is None:
        return
    course_plan.status = "generated"
    await session.commit()


async def list_pending(
    session: AsyncSession, *, page: int, limit: int, now: datetime
) -> tuple[list[CoursePlan], int]:
    """Plans `pending` non expirés (plus récents d'abord), paginés."""
    condition = (CoursePlan.status == "pending", CoursePlan.expires_at > now)

    total = (await session.execute(select(func.count(CoursePlan.id)).where(*condition))).scalar_one()
    result = await session.execute(
        select(CoursePlan)
        .where(*condition)
        .order_by(CoursePlan.created_at.desc())
        .offset((page - 1) * limit)
        .limit(limit)
    )
    return list(result.scalars().all()), total
