"""Repository for video_generation_jobs table.

Follows the same async pattern as course_session_repository.py:
functions receive an AsyncSession, commit + refresh on write.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import VideoGenerationJob


async def create(
    session: AsyncSession,
    *,
    course_session_id: uuid.UUID,
) -> VideoGenerationJob:
    """Crée un job de génération vidéo en statut pending."""
    job = VideoGenerationJob(
        course_session_id=course_session_id,
        status="pending",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)
    return job


async def get_by_id(
    session: AsyncSession,
    job_id: uuid.UUID,
) -> VideoGenerationJob | None:
    """Récupère un job par son UUID."""
    result = await session.execute(
        select(VideoGenerationJob).where(VideoGenerationJob.id == job_id)
    )
    return result.scalar_one_or_none()


async def get_active_job_for_session(
    session: AsyncSession,
    course_session_id: uuid.UUID,
) -> VideoGenerationJob | None:
    """Vérifie si un job pending/running existe déjà pour une session donnée."""
    result = await session.execute(
        select(VideoGenerationJob).where(
            VideoGenerationJob.course_session_id == course_session_id,
            VideoGenerationJob.status.in_(["pending", "running"]),
        )
    )
    return result.scalars().first()


async def update_status(
    session: AsyncSession,
    job_id: uuid.UUID,
    *,
    status: str,
    model_used: str | None = None,
    fallback_used: bool | None = None,
    video_path: str | None = None,
    error_message: str | None = None,
) -> VideoGenerationJob | None:
    """Met à jour le statut et les champs d'un job vidéo."""
    job = await get_by_id(session, job_id)
    if job is None:
        return None

    job.status = status
    job.updated_at = datetime.now(timezone.utc)
    if model_used is not None:
        job.model_used = model_used
    if fallback_used is not None:
        job.fallback_used = fallback_used
    if video_path is not None:
        job.video_path = video_path
    if error_message is not None:
        job.error_message = error_message

    await session.commit()
    await session.refresh(job)
    return job
