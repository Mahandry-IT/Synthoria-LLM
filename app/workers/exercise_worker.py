"""Worker de génération des exercices corrigés : lit la file `exercise_jobs` et exécute un job à la fois.

Lancement : python -m app.workers.exercise_worker
Plusieurs workers peuvent tourner en parallèle (réclamation FOR UPDATE SKIP LOCKED).
"""

import asyncio
import logging
import sys

from sqlalchemy.ext.asyncio import async_sessionmaker

from app.core.config import Settings, get_settings
from app.db.session import create_engine
from app.repositories import exercise_job_repository as jobs_repo
from app.services.exercise_jobs import run_exercise_job
from app.services.gemini_client import GeminiClient
from app.workers.signals import install_signal_handlers

logger = logging.getLogger(__name__)


async def process_next_job(
    session_factory: async_sessionmaker,
    gemini_client: GeminiClient,
    settings: Settings,
    should_stop=lambda: False,
) -> bool:
    """Libère les jobs périmés puis réclame et exécute au plus un job. True si un job a été traité."""
    async with session_factory() as db:
        await jobs_repo.release_stale(
            db, stale_minutes=settings.exercise_job_stale_minutes, max_attempts=settings.exercise_max_attempts
        )
    async with session_factory() as db:
        job = await jobs_repo.claim_next(db)
    if job is None:
        return False
    outcome = await run_exercise_job(
        job, session_factory=session_factory, gemini_client=gemini_client, settings=settings, should_stop=should_stop
    )
    logger.info("exercise_job_finished", extra={"job_id": str(job.id), "outcome": outcome})
    return True


async def run_worker(settings: Settings, stop_event: asyncio.Event) -> None:
    engine, session_factory = create_engine(settings)
    # session_factory injecté : quota Gemini partagé avec l'api et le worker podcast (gemini_quota_manager).
    gemini_client = GeminiClient(settings, session_factory=session_factory)
    logger.info("exercise_worker_started")
    try:
        while not stop_event.is_set():
            try:
                processed = await process_next_job(
                    session_factory, gemini_client, settings, should_stop=stop_event.is_set
                )
            except Exception:  # noqa: BLE001 — le worker ne doit jamais mourir sur une erreur de boucle
                logger.error("exercise_worker_loop_error", exc_info=True)
                processed = False
            if not processed:
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=settings.exercise_worker_poll_seconds)
                except asyncio.TimeoutError:
                    pass
    finally:
        await engine.dispose()
        logger.info("exercise_worker_stopped")


async def _main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)], force=True,
    )
    stop_event = asyncio.Event()
    install_signal_handlers(asyncio.get_running_loop(), stop_event)
    await run_worker(get_settings(), stop_event)


if __name__ == "__main__":
    asyncio.run(_main())
