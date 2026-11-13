"""Mise en file idempotente et exécution du job de génération des exercices (un lot par cours, RG-9)."""

import hashlib
import json
import logging
import uuid

from sqlalchemy.ext.asyncio import async_sessionmaker

from app.core.config import Settings
from app.core.exceptions import (
    GeminiDailyQuotaExceededError,
    GeminiInvalidResponseError,
    GeminiQuotaExceededError,
    GeminiServiceError,
)
from app.db.models import ExerciseJob
from app.repositories import course_session_repository, exercise_job_repository as jobs_repo, exercise_repository
from app.services.course_depth import depth_of_session
from app.services.exercise_generator import generate_exercises, max_exercises
from app.services.gemini_client import GeminiClient

logger = logging.getLogger(__name__)

# Messages fixes montrés à l'apprenant : jamais le texte brut d'une erreur Gemini.
MSG_QUOTA_DAILY = "Le quota Gemini du jour est atteint : réessayez plus tard."
MSG_EXHAUSTED = "La génération des exercices a échoué après plusieurs tentatives : réessayez."
MSG_COURSE_GONE = "Le cours n'existe plus."
_MSG_RETRYING = "Nouvelle tentative en cours."


class CourseSessionNotFoundError(Exception):
    """Aucune session de cours ne correspond à l'id fourni."""


def compute_params_hash(depth: str, cap: int) -> str:
    """Empreinte des paramètres de génération (mode et plafond), pour tracer ce qui a produit le lot."""
    payload = json.dumps({"depth": depth, "cap": cap}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


async def enqueue_exercise_job(
    session_factory: async_sessionmaker, session_id: uuid.UUID, gemini_response: dict, settings: Settings
) -> ExerciseJob:
    """Retourne le job du cours, en le créant s'il n'existe pas ou en le relançant s'il est `failed`.

    Idempotent : un job `pending`, `generating` ou `done` est renvoyé tel quel, jamais régénéré.
    """
    depth = depth_of_session(gemini_response)
    params_hash = compute_params_hash(depth, max_exercises(depth, settings))
    async with session_factory() as db:
        job = await jobs_repo.get_by_session(db, session_id)
        if job is None:
            await jobs_repo.create_if_absent(db, session_id=session_id, params_hash=params_hash)
        elif job.status == "failed":
            await jobs_repo.restart_failed(db, session_id, params_hash=params_hash)
        else:
            return job
        job = await jobs_repo.get_by_session(db, session_id)
    if job is None:  # cours supprimé entre-temps (cascade)
        raise CourseSessionNotFoundError(str(session_id))
    return job


def _retry_delay(exc: Exception, settings: Settings) -> float:
    hinted = getattr(exc, "retry_after_seconds", None)
    return max(float(hinted), settings.exercise_retry_delay_seconds) if hinted else settings.exercise_retry_delay_seconds


async def _fail_or_requeue(
    session_factory: async_sessionmaker, job: ExerciseJob, settings: Settings, *, delay: float
) -> str:
    async with session_factory() as db:
        if job.attempts >= settings.exercise_max_attempts:
            await jobs_repo.mark_failed(db, job.id, MSG_EXHAUSTED)
            return "failed"
        await jobs_repo.requeue(db, job.id, error_message=_MSG_RETRYING, delay_seconds=delay)
        return "requeued"


async def _requeue_rate_limited(
    session_factory: async_sessionmaker, job: ExerciseJob, settings: Settings, exc: Exception
) -> str:
    """429 « minute » : l'échec ne vient pas du job, la tentative est remboursée (elle ne compte pas
    dans `exercise_max_attempts`) et la reprise attend le délai indiqué par Google."""
    async with session_factory() as db:
        await jobs_repo.requeue(
            db, job.id, error_message=_MSG_RETRYING, refund_attempt=True, delay_seconds=_retry_delay(exc, settings)
        )
    return "requeued"


async def run_exercise_job(
    job: ExerciseJob,
    *,
    session_factory: async_sessionmaker,
    gemini_client: GeminiClient,
    settings: Settings,
    should_stop=lambda: False,
) -> str:
    """Exécute un job réclamé (`generating`) : un appel Gemini, puis exercices et statut `done` dans UNE
    transaction (jamais de lot partiel). Retourne : done, failed, requeued, interrupted ou lost (job repris par un autre)."""
    if should_stop():
        async with session_factory() as db:
            await jobs_repo.requeue(db, job.id, refund_attempt=True)
        return "interrupted"

    async with session_factory() as db:
        course = await course_session_repository.get_by_id(db, job.session_id)
    if course is None:
        async with session_factory() as db:
            await jobs_repo.mark_failed(db, job.id, MSG_COURSE_GONE)
        return "failed"

    try:
        rows = await generate_exercises(course.gemini_response, gemini_client, settings)
    except GeminiDailyQuotaExceededError:
        # Aucun retry ne peut réussir avant le reset du quota : échec net, relançable par l'apprenant.
        async with session_factory() as db:
            await jobs_repo.mark_failed(db, job.id, MSG_QUOTA_DAILY)
        return "failed"
    except GeminiQuotaExceededError as exc:
        logger.warning("exercise_job_rate_limited", extra={"job_id": str(job.id)})
        return await _requeue_rate_limited(session_factory, job, settings, exc)
    except (GeminiServiceError, GeminiInvalidResponseError) as exc:
        # Indisponibilité, réponse invalide : nouvelle tentative avec attente, bornée.
        logger.warning("exercise_job_retry", extra={"job_id": str(job.id), "error": type(exc).__name__})
        return await _fail_or_requeue(session_factory, job, settings, delay=_retry_delay(exc, settings))
    except Exception:  # noqa: BLE001 — un job ne doit jamais rester bloqué en `generating`
        logger.error("exercise_job_error", extra={"job_id": str(job.id)}, exc_info=True)
        return await _fail_or_requeue(session_factory, job, settings, delay=settings.exercise_retry_delay_seconds)

    async with session_factory() as db:
        # `done` d'abord (UPDATE gardé sur `generating`) : si le job n'est plus à nous, aucun exercice écrit.
        if not await jobs_repo.mark_done(db, job.id):
            logger.warning("exercise_job_lost", extra={"job_id": str(job.id)})
            return "lost"
        await exercise_repository.replace_all(db, job.session_id, [r.as_row() for r in rows])
        await db.commit()
    return "done"
