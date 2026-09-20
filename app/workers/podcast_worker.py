"""Worker de jobs podcast : lit la file (table podcast_jobs) et exécute le pipeline.

Lancement : python -m app.workers.podcast_worker
Plusieurs workers peuvent tourner en parallèle (réclamation FOR UPDATE SKIP LOCKED).
"""

import asyncio
import logging
import shutil
import signal
import sys
import time
from pathlib import Path

from sqlalchemy.ext.asyncio import async_sessionmaker

from app.core.config import Settings, get_settings
from app.db.session import create_engine
from app.repositories import podcast_job_repository as jobs_repo
from app.services.gemini_client import GeminiClient
from app.services.podcast.pipeline import run_podcast_job
from app.services.podcast.tts_client import PiperHTTPEngine, TTSEngine

logger = logging.getLogger(__name__)

PURGE_INTERVAL_SECONDS = 3600


def purge_expired_dirs(settings: Settings, *, now: float | None = None) -> int:
    """Supprime les dossiers de jobs plus vieux que la rétention. Retourne le nombre supprimé."""
    root = Path(settings.podcast_storage_dir)
    if not root.is_dir() or settings.podcast_retention_days <= 0:
        return 0
    threshold = (now if now is not None else time.time()) - settings.podcast_retention_days * 86400
    removed = 0
    for child in root.iterdir():
        if child.is_dir() and child.stat().st_mtime < threshold:
            shutil.rmtree(child, ignore_errors=True)
            removed += 1
    return removed


async def process_next_job(
    session_factory: async_sessionmaker,
    gemini_client: GeminiClient,
    tts_engine: TTSEngine,
    settings: Settings,
    should_stop=lambda: False,
) -> bool:
    """Libère les jobs périmés puis réclame et exécute au plus un job. True si un job a été traité."""
    async with session_factory() as db:
        await jobs_repo.release_stale(
            db, stale_minutes=settings.podcast_job_stale_minutes, max_attempts=settings.podcast_max_attempts
        )
    async with session_factory() as db:
        job = await jobs_repo.claim_next(db)
    if job is None:
        return False
    outcome = await run_podcast_job(
        job.id, session_factory=session_factory, gemini_client=gemini_client,
        tts_engine=tts_engine, settings=settings, should_stop=should_stop,
    )
    logger.info("podcast_job_finished", extra={"job_id": str(job.id), "outcome": outcome})
    return True


async def run_worker(settings: Settings, stop_event: asyncio.Event) -> None:
    engine, session_factory = create_engine(settings)
    gemini_client = GeminiClient(settings)
    tts_engine = PiperHTTPEngine.from_settings(settings)
    last_purge = 0.0
    logger.info("podcast_worker_started")
    try:
        while not stop_event.is_set():
            if time.monotonic() - last_purge > PURGE_INTERVAL_SECONDS:
                last_purge = time.monotonic()
                logger.info("podcast_purge", extra={"removed": purge_expired_dirs(settings)})
            try:
                processed = await process_next_job(
                    session_factory, gemini_client, tts_engine, settings, should_stop=stop_event.is_set
                )
            except Exception:  # noqa: BLE001 — le worker ne doit jamais mourir sur une erreur de boucle
                logger.error("podcast_worker_loop_error", exc_info=True)
                processed = False
            if not processed:
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=settings.podcast_worker_poll_seconds)
                except asyncio.TimeoutError:
                    pass
    finally:
        await tts_engine.close()
        await engine.dispose()
        logger.info("podcast_worker_stopped")


def _install_signal_handlers(loop: asyncio.AbstractEventLoop, stop_event: asyncio.Event) -> None:
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:  # Windows
            signal.signal(sig, lambda *_: loop.call_soon_threadsafe(stop_event.set))


async def _main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)], force=True,
    )
    stop_event = asyncio.Event()
    _install_signal_handlers(asyncio.get_running_loop(), stop_event)
    await run_worker(get_settings(), stop_event)


if __name__ == "__main__":
    asyncio.run(_main())
