"""File de génération des exercices (`exercise_jobs`) : une ligne par cours, modèle du podcast.

Sur un job `pending`, `locked_at` est un « pas avant » (attente après un 429) ; sur un job `generating`,
c'est le verrou du worker, périmé au-delà de `stale_minutes`.
"""

import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import Select, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ExerciseJob

ACTIVE_STATUS = "generating"


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def get_by_session(session: AsyncSession, session_id: uuid.UUID) -> ExerciseJob | None:
    result = await session.execute(select(ExerciseJob).where(ExerciseJob.session_id == session_id))
    return result.scalar_one_or_none()


async def get_by_id(session: AsyncSession, job_id: uuid.UUID) -> ExerciseJob | None:
    result = await session.execute(select(ExerciseJob).where(ExerciseJob.id == job_id))
    return result.scalar_one_or_none()


async def create_if_absent(session: AsyncSession, *, session_id: uuid.UUID, params_hash: str) -> bool:
    """Crée le job du cours s'il n'existe pas (INSERT ... ON CONFLICT DO NOTHING : deux lancements
    simultanés ne produisent qu'une ligne). Retourne True si cet appel l'a créé."""
    result = await session.execute(
        pg_insert(ExerciseJob)
        .values(id=uuid.uuid4(), session_id=session_id, params_hash=params_hash, status="pending", attempts=0,
                created_at=_now(), updated_at=_now())
        .on_conflict_do_nothing(index_elements=[ExerciseJob.session_id])
        .returning(ExerciseJob.id)
    )
    created = result.scalar_one_or_none() is not None
    await session.commit()
    return created


async def restart_failed(session: AsyncSession, session_id: uuid.UUID, *, params_hash: str) -> bool:
    """Relance un job `failed` (seul état relançable) : UPDATE conditionnel, atomique. True si relancé."""
    result = await session.execute(
        update(ExerciseJob)
        .where(ExerciseJob.session_id == session_id, ExerciseJob.status == "failed")
        .values(status="pending", attempts=0, locked_at=None, error_message=None, params_hash=params_hash,
                updated_at=_now())
    )
    await session.commit()
    return bool(result.rowcount)


def claim_statement() -> Select:
    """Plus ancien job `pending` dont l'attente est écoulée, verrouillé sans bloquer les autres workers."""
    return (
        select(ExerciseJob)
        .where(
            ExerciseJob.status == "pending",
            or_(ExerciseJob.locked_at.is_(None), ExerciseJob.locked_at <= func.now()),
        )
        .order_by(ExerciseJob.created_at)
        .limit(1)
        .with_for_update(skip_locked=True)
    )


async def claim_next(session: AsyncSession) -> ExerciseJob | None:
    """Réclame le prochain job (FOR UPDATE SKIP LOCKED) : deux workers ne prennent jamais le même."""
    result = await session.execute(claim_statement())
    job = result.scalar_one_or_none()
    if job is None:
        await session.rollback()
        return None
    job.status = ACTIVE_STATUS
    job.locked_at = _now()
    job.attempts = (job.attempts or 0) + 1
    job.error_message = None
    await session.commit()
    await session.refresh(job)
    return job


async def release_stale(session: AsyncSession, *, stale_minutes: int, max_attempts: int) -> int:
    """Libère les jobs `generating` dont le worker a disparu : retour à `pending` sous `max_attempts`,
    sinon `failed`. Retourne le nombre de jobs touchés."""
    threshold = _now() - timedelta(minutes=stale_minutes)
    stale = (ExerciseJob.status == ACTIVE_STATUS) & (ExerciseJob.locked_at < threshold)
    exhausted = await session.execute(
        update(ExerciseJob)
        .where(stale, ExerciseJob.attempts >= max_attempts)
        .values(status="failed", locked_at=None, error_message="Nombre maximal de tentatives atteint")
    )
    retried = await session.execute(
        update(ExerciseJob).where(stale, ExerciseJob.attempts < max_attempts).values(status="pending", locked_at=None)
    )
    await session.commit()
    return (exhausted.rowcount or 0) + (retried.rowcount or 0)


async def mark_done(session: AsyncSession, job_id: uuid.UUID) -> bool:
    """Passe le job `generating` à `done` dans la transaction courante, SANS valider : à appeler AVANT
    d'ajouter les exercices, puis l'appelant valide d'un seul commit (jamais de lot partiel).
    Si le job n'est plus à nous (libéré comme périmé, supprimé...), rien n'est modifié : la transaction
    est annulée et la fonction retourne False, l'appelant n'écrit alors aucun exercice."""
    result = await session.execute(
        update(ExerciseJob)
        .where(ExerciseJob.id == job_id, ExerciseJob.status == ACTIVE_STATUS)
        .values(status="done", locked_at=None, error_message=None)
    )
    if not result.rowcount:
        await session.rollback()
        return False
    return True


async def mark_failed(session: AsyncSession, job_id: uuid.UUID, error_message: str) -> None:
    await session.execute(
        update(ExerciseJob)
        .where(ExerciseJob.id == job_id)
        .values(status="failed", error_message=error_message[:2000], locked_at=None)
    )
    await session.commit()


async def requeue(
    session: AsyncSession,
    job_id: uuid.UUID,
    *,
    error_message: str | None = None,
    refund_attempt: bool = False,
    delay_seconds: float = 0,
) -> None:
    """Remet le job en file. `delay_seconds` : pas de reprise avant ce délai (429) ;
    `refund_attempt` : ne compte pas la tentative (arrêt propre du worker)."""
    # Même source d'horloge que la comparaison de claim_statement (func.now() côté base) : pas de décalage
    # entre l'horloge du worker et celle de Postgres.
    values: dict = {"status": "pending", "locked_at": func.now() + timedelta(seconds=delay_seconds) if delay_seconds else None}
    if error_message is not None:
        values["error_message"] = error_message[:2000]
    if refund_attempt:
        values["attempts"] = func.greatest(ExerciseJob.attempts - 1, 0)
    await session.execute(update(ExerciseJob).where(ExerciseJob.id == job_id).values(**values))
    await session.commit()
