"""Orchestrateur du pipeline podcast : cours persisté → script → voix → MP3.

Étapes : A sérialisation (pure) → B script Gemini [checkpoint DB] → C/D normalisation
et synthèse [cache disque par hash] → E assemblage ffmpeg. Une reprise saute ce qui est
déjà fait : le script en base n'est jamais régénéré, les WAV en cache jamais resynthétisés.
"""

import logging
import shutil
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.core.config import Settings
from app.core.exceptions import TTSInvalidVoiceError
from app.repositories import course_session_repository, podcast_job_repository as jobs_repo
from app.schemas.podcast import PodcastScript
from app.services.gemini_client import GeminiClient
from app.services.podcast.audio_assembler import TimedTurn, assemble_mp3, build_timeline, render_vtt
from app.services.podcast.course_serializer import has_usable_content, serialize_course
from app.services.podcast.jobs import job_dir
from app.services.podcast.script_generator import ScriptRequest, generate_script
from app.services.podcast.tts_client import TTSEngine, synthesize_many
from app.services.podcast.tts_normalizer import normalize_for_tts, split_for_tts

logger = logging.getLogger(__name__)

AUDIO_FILENAME = "podcast.mp3"
TRANSCRIPT_FILENAME = "transcript.vtt"


class JobNotFoundError(Exception):
    """Le job demandé n'existe pas."""


class JobInterrupted(Exception):
    """Arrêt propre demandé (SIGTERM) : le job est remis en file sans compter de tentative."""


class PermanentJobError(Exception):
    """Échec définitif (inutile de réessayer)."""


@dataclass(frozen=True)
class SpokenTurn:
    speaker: str
    text: str
    chapter: str | None
    think_pause: bool = False


def build_spoken_turns(script: PodcastScript) -> list[SpokenTurn]:
    """Aplatit le script en répliques ordonnées ; la première réplique de chaque bloc ouvre un chapitre."""
    spoken: list[SpokenTurn] = []
    for index, turn in enumerate(script.intro_turns):
        spoken.append(SpokenTurn(turn.speaker, turn.text, "Introduction" if index == 0 else None, turn.think_pause))
    for segment in script.segments:
        for index, turn in enumerate(segment.turns):
            spoken.append(
                SpokenTurn(turn.speaker, turn.text, segment.title if index == 0 else None, turn.think_pause)
            )
    for index, turn in enumerate(script.outro_turns):
        spoken.append(SpokenTurn(turn.speaker, turn.text, "Conclusion" if index == 0 else None))
    return spoken


def voices_for(settings: Settings) -> dict[str, str]:
    return {"HOST": settings.podcast_voice_host, "EXPERT": settings.podcast_voice_expert}


@asynccontextmanager
async def _stage(job_id: uuid.UUID, name: str):
    logger.info("podcast_stage_started", extra={"job_id": str(job_id), "stage": name})
    started = time.monotonic()
    yield
    logger.info(
        "podcast_stage_finished",
        extra={"job_id": str(job_id), "stage": name, "duration_ms": round((time.monotonic() - started) * 1000)},
    )


async def run_podcast_job(
    job_id: uuid.UUID,
    *,
    session_factory: async_sessionmaker,
    gemini_client: GeminiClient,
    tts_engine: TTSEngine,
    settings: Settings,
    should_stop: Callable[[], bool] = lambda: False,
) -> str:
    """Exécute (ou reprend) un job. Retourne l'état final : done, failed, requeued ou interrupted."""
    async with session_factory() as db:
        job = await jobs_repo.get_by_id(db, job_id)
        if job is None:
            raise JobNotFoundError(str(job_id))
        course_row = await course_session_repository.get_by_id(db, job.course_session_id)
        params = dict(job.params or {})
        checkpoint = job.script
        attempts = job.attempts or 0

    async def set_stage(status: str, stage: str, progress: int) -> None:
        async with session_factory() as db:
            await jobs_repo.update_stage(db, job_id, status=status, stage=stage, progress=progress)

    def checkpoint_stop() -> None:
        if should_stop():
            raise JobInterrupted()

    try:
        if course_row is None:
            raise PermanentJobError("Session de cours introuvable")

        # A. Sérialisation déterministe.
        course = course_row.gemini_response
        sections = serialize_course(course)
        if not has_usable_content(sections):
            raise PermanentJobError("Le cours ne contient aucun contenu exploitable")

        # B. Script (checkpoint en base).
        script: PodcastScript | None = None
        if checkpoint:
            try:
                script = PodcastScript.model_validate(checkpoint)
                logger.info("podcast_script_reused", extra={"job_id": str(job_id)})
            except ValidationError:
                logger.warning("podcast_script_checkpoint_invalid", extra={"job_id": str(job_id)})
        if script is None:
            await set_stage("scripting", "scripting", 5)
            async with _stage(job_id, "scripting"):
                script = await generate_script(
                    sections,
                    ScriptRequest(
                        course_title=course["meta"]["title"],
                        style=str(params.get("style", "conversational")),
                        target_minutes=int(params.get("target_minutes", settings.podcast_default_target_minutes)),
                    ),
                    gemini_client,
                    settings,
                )
            async with session_factory() as db:
                await jobs_repo.save_script(db, job_id, script.model_dump(mode="json"))
        checkpoint_stop()

        # C/D. Normalisation + synthèse (cache disque).
        directory = job_dir(settings, job_id)
        cache_dir, work_dir = directory / "cache", directory / "work"
        directory.mkdir(parents=True, exist_ok=True)
        voices = voices_for(settings)

        await set_stage("synthesizing", "synthesizing", 30)
        kept: list[tuple[SpokenTurn, list[str]]] = []
        pending_chapter: str | None = None
        for turn in build_spoken_turns(script):
            pending_chapter = turn.chapter or pending_chapter
            chunks = split_for_tts(normalize_for_tts(turn.text), settings.podcast_tts_max_chars)
            if not chunks:
                continue
            kept.append((SpokenTurn(turn.speaker, turn.text, pending_chapter, turn.think_pause), chunks))
            pending_chapter = None
        if not kept:
            raise PermanentJobError("Le script ne contient aucune réplique prononçable")

        items = [(voices.get(turn.speaker, voices["EXPERT"]), chunk) for turn, chunks in kept for chunk in chunks]
        last_reported = {"value": 30}

        async def on_progress(done: int, total: int) -> None:
            progress = 30 + round(55 * done / total)
            if progress - last_reported["value"] >= 5 or done == total:
                last_reported["value"] = progress
                await set_stage("synthesizing", "synthesizing", progress)

        async with _stage(job_id, "synthesizing"):
            paths = await synthesize_many(
                tts_engine, cache_dir, items, concurrency=settings.podcast_tts_concurrency, on_progress=on_progress
            )
        checkpoint_stop()

        # E. Assemblage.
        await set_stage("mixing", "mixing", 88)
        timed: list[TimedTurn] = []
        cursor = 0
        for turn, chunks in kept:
            timed.append(
                TimedTurn(
                    turn.speaker, turn.text, paths[cursor : cursor + len(chunks)], turn.chapter, turn.think_pause
                )
            )
            cursor += len(chunks)
        timeline = build_timeline(timed)
        output = directory / AUDIO_FILENAME
        async with _stage(job_id, "mixing"):
            await assemble_mp3(
                timeline, title=script.title, output_path=output, work_dir=work_dir,
                bitrate=settings.podcast_audio_bitrate,
            )
        (directory / TRANSCRIPT_FILENAME).write_text(render_vtt(timeline.cues), encoding="utf-8")

        async with session_factory() as db:
            await jobs_repo.mark_done(
                db, job_id, audio_path=str(output), duration_seconds=timeline.duration,
                tts_engine=tts_engine.name, voices=voices,
            )
        shutil.rmtree(cache_dir, ignore_errors=True)
        shutil.rmtree(work_dir, ignore_errors=True)
        return "done"

    except JobInterrupted:
        async with session_factory() as db:
            await jobs_repo.requeue(db, job_id, refund_attempt=True)
        return "interrupted"
    except (PermanentJobError, TTSInvalidVoiceError) as exc:
        return await _fail(session_factory, job_id, str(exc))
    except Exception as exc:  # noqa: BLE001 — toute erreur d'étape est tracée sur le job
        logger.error("podcast_job_failed", extra={"job_id": str(job_id), "error": type(exc).__name__}, exc_info=True)
        message = f"{type(exc).__name__}: {exc}"
        if attempts < settings.podcast_max_attempts:
            async with session_factory() as db:
                await jobs_repo.requeue(db, job_id, error_message=message)
            return "requeued"
        return await _fail(session_factory, job_id, message)


async def _fail(session_factory: async_sessionmaker, job_id: uuid.UUID, message: str) -> str:
    async with session_factory() as db:
        await jobs_repo.mark_failed(db, job_id, message)
    return "failed"


def audio_file(settings: Settings, job_id: uuid.UUID) -> Path:
    return job_dir(settings, job_id) / AUDIO_FILENAME


def transcript_file(settings: Settings, job_id: uuid.UUID) -> Path:
    return job_dir(settings, job_id) / TRANSCRIPT_FILENAME

