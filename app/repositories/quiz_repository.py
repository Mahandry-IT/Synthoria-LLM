"""Accès aux tables `quiz_questions` (banque de QCM) et `quiz_attempts` (tentatives notées)."""

import uuid
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import CourseSession, QuizAttempt, QuizBankQuestion


async def lock_session(session: AsyncSession, session_id: uuid.UUID) -> CourseSession | None:
    """Session verrouillée (`FOR UPDATE`) jusqu'au commit : sérialise l'amorçage de la banque et
    la mise à jour des compteurs de service quand deux tentatives démarrent en même temps."""
    result = await session.execute(select(CourseSession).where(CourseSession.id == session_id).with_for_update())
    return result.scalar_one_or_none()


async def session_exists(session: AsyncSession, session_id: uuid.UUID) -> bool:
    result = await session.execute(select(CourseSession.id).where(CourseSession.id == session_id))
    return result.scalar_one_or_none() is not None


async def list_bank(session: AsyncSession, session_id: uuid.UUID) -> list[QuizBankQuestion]:
    result = await session.execute(
        select(QuizBankQuestion).where(QuizBankQuestion.session_id == session_id).order_by(QuizBankQuestion.created_at)
    )
    return list(result.scalars().all())


async def max_batch(session: AsyncSession, session_id: uuid.UUID) -> int:
    result = await session.execute(
        select(func.max(QuizBankQuestion.batch)).where(QuizBankQuestion.session_id == session_id)
    )
    value = result.scalar_one_or_none()
    return value if value is not None else -1


def add_questions(
    session: AsyncSession, session_id: uuid.UUID, payloads: list[dict], batch: int
) -> list[QuizBankQuestion]:
    """Ajoute des questions à la banque (sans commit : l'appelant contrôle la transaction)."""
    rows = [QuizBankQuestion(id=uuid.uuid4(), session_id=session_id, payload=p, batch=batch, times_served=0) for p in payloads]
    session.add_all(rows)
    return rows


async def get_questions(session: AsyncSession, question_ids: list[uuid.UUID]) -> dict[uuid.UUID, QuizBankQuestion]:
    if not question_ids:
        return {}
    result = await session.execute(select(QuizBankQuestion).where(QuizBankQuestion.id.in_(question_ids)))
    return {q.id: q for q in result.scalars().all()}


def create_attempt(
    session: AsyncSession, session_id: uuid.UUID, question_ids: list[uuid.UUID], max_score: float, now: datetime
) -> QuizAttempt:
    attempt = QuizAttempt(
        id=uuid.uuid4(),
        session_id=session_id,
        question_ids=[str(q) for q in question_ids],
        max_score=max_score,
        status="in_progress",
        started_at=now,
    )
    session.add(attempt)
    return attempt


async def get_attempt_for_update(
    session: AsyncSession, session_id: uuid.UUID, attempt_id: uuid.UUID
) -> QuizAttempt | None:
    """Tentative verrouillée : deux soumissions concurrentes ne peuvent pas noter deux fois (409)."""
    result = await session.execute(
        select(QuizAttempt)
        .where(QuizAttempt.id == attempt_id, QuizAttempt.session_id == session_id)
        .with_for_update()
    )
    return result.scalar_one_or_none()


async def list_attempts(session: AsyncSession, session_id: uuid.UUID, limit: int) -> list[QuizAttempt]:
    result = await session.execute(
        select(QuizAttempt)
        .where(QuizAttempt.session_id == session_id)
        .order_by(QuizAttempt.started_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())
