import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import Select, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import PodcastJob

ACTIVE_STATUSES = ("scripting", "synthesizing", "mixing")


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def create(
    session: AsyncSession,
    *,
    course_session_id: uuid.UUID,
    params: dict[str, Any],
    params_hash: str,
) -> PodcastJob:
    """Crée un job pending."""
    job = PodcastJob(
        course_session_id=course_session_id,
        params=params,
        params_hash=params_hash,
        status="pending",
        progress=0,
        attempts=0,
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)
    return job


async def get_by_id(session: AsyncSession, job_id: uuid.UUID) -> PodcastJob | None:
    result = await session.execute(select(PodcastJob).where(PodcastJob.id == job_id))
    return result.scalar_one_or_none()


async def find_reusable(
    session: AsyncSession, course_session_id: uuid.UUID, params_hash: str
) -> PodcastJob | None:
    """Job le plus récent non échoué pour (session, paramètres) — base de l'idempotence."""
    result = await session.execute(
        select(PodcastJob)
        .where(
            PodcastJob.course_session_id == course_session_id,
            PodcastJob.params_hash == params_hash,
            PodcastJob.status != "failed",
        )
        .order_by(PodcastJob.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def list_by_session(session: AsyncSession, course_session_id: uuid.UUID) -> list[PodcastJob]:
    result = await session.execute(
        select(PodcastJob)
        .where(PodcastJob.course_session_id == course_session_id)
        .order_by(PodcastJob.created_at.desc())
    )
    return list(result.scalars().all())


def claim_statement() -> Select:
    """Requête de réclamation : le plus ancien job pending, verrouillé sans bloquer les autres workers."""
    return (
        select(PodcastJob)
        .where(PodcastJob.status == "pending")
        .order_by(PodcastJob.created_at)
        .limit(1)
        .with_for_update(skip_locked=True)
    )


async def claim_next(session: AsyncSession) -> PodcastJob | None:
    """Réclame le prochain job (FOR UPDATE SKIP LOCKED) : deux workers ne prennent jamais le même."""
    result = await session.execute(claim_statement())
    job = result.scalar_one_or_none()
    if job is None:
        await session.rollback()
        return None
    job.status = "scripting"
    job.locked_at = _now()
    job.attempts = (job.attempts or 0) + 1
    job.error_message = None
    await session.commit()
    await session.refresh(job)
    return job


async def release_stale(session: AsyncSession, *, stale_minutes: int, max_attempts: int) -> int:
    """Libère les jobs dont le worker a disparu (verrou périmé).

    Sous max_attempts → retour à pending (reprise depuis le dernier checkpoint) ;
    sinon → failed définitif. Retourne le nombre de jobs touchés.
    """
    threshold = _now() - timedelta(minutes=stale_minutes)
    stale = PodcastJob.status.in_(ACTIVE_STATUSES) & (PodcastJob.locked_at < threshold)

    exhausted = await session.execute(
        update(PodcastJob)
        .where(stale, PodcastJob.attempts >= max_attempts)
        .values(status="failed", locked_at=None, error_message="Nombre maximal de tentatives atteint")
    )
    retried = await session.execute(
        update(PodcastJob)
        .where(stale, PodcastJob.attempts < max_attempts)
        .values(status="pending", locked_at=None)
    )
    await session.commit()
    return (exhausted.rowcount or 0) + (retried.rowcount or 0)


async def update_stage(
    session: AsyncSession, job_id: uuid.UUID, *, status: str, stage: str | None, progress: int
) -> None:
    """Met à jour l'étape courante et renouvelle le verrou (heartbeat)."""
    await session.execute(
        update(PodcastJob)
        .where(PodcastJob.id == job_id)
        .values(status=status, stage=stage, progress=progress, locked_at=_now())
    )
    await session.commit()


async def save_script(session: AsyncSession, job_id: uuid.UUID, script: dict[str, Any]) -> None:
    """Checkpoint de l'étape de scriptage."""
    await session.execute(update(PodcastJob).where(PodcastJob.id == job_id).values(script=script))
    await session.commit()


async def mark_done(
    session: AsyncSession,
    job_id: uuid.UUID,
    *,
    audio_path: str,
    duration_seconds: float,
    tts_engine: str,
    voices: dict[str, str],
) -> None:
    await session.execute(
        update(PodcastJob)
        .where(PodcastJob.id == job_id)
        .values(
            status="done", stage="done", progress=100, audio_path=audio_path,
            duration_seconds=duration_seconds, tts_engine=tts_engine, voices=voices,
            locked_at=None, error_message=None,
        )
    )
    await session.commit()


async def mark_failed(session: AsyncSession, job_id: uuid.UUID, error_message: str) -> None:
    await session.execute(
        update(PodcastJob)
        .where(PodcastJob.id == job_id)
        .values(status="failed", error_message=error_message[:2000], locked_at=None)
    )
    await session.commit()
