"""CLI d'automatisation (stdlib argparse) : python -m app.cli podcast <enqueue|run|status> ...

Codes de sortie : 0 succès, 1 erreur/échec du job, 2 ressource introuvable.
"""

import argparse
import asyncio
import json
import logging
import sys
import uuid
from typing import Any

from app.api.schemas import PodcastGenerationRequest
from app.core.config import Settings, get_settings
from app.db.session import create_engine
from app.repositories import podcast_job_repository as jobs_repo
from app.services.gemini_client import GeminiClient
from app.services.podcast.jobs import CourseSessionNotFoundError, enqueue_podcast_job
from app.services.podcast.pipeline import JobNotFoundError, run_podcast_job
from app.services.podcast.tts_client import PiperHTTPEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="app.cli", description="Outils en ligne de commande Synthoria")
    groups = parser.add_subparsers(dest="group", required=True)
    podcast = groups.add_parser("podcast", help="Gestion des podcasts")
    commands = podcast.add_subparsers(dest="command", required=True)

    enqueue = commands.add_parser("enqueue", help="Met un job en file pour une session de cours")
    enqueue.add_argument("session_id", type=uuid.UUID)
    enqueue.add_argument("--force", action="store_true", help="Crée un nouveau job même s'il en existe un")
    enqueue.add_argument("--style", choices=["conversational", "educational", "concise"], default="conversational")
    enqueue.add_argument("--minutes", type=int, default=None, help="Durée cible (3-60)")

    run = commands.add_parser("run", help="Exécute un job de façon synchrone, sans worker")
    run.add_argument("job_id", type=uuid.UUID)

    status = commands.add_parser("status", help="Affiche l'état d'un job (JSON)")
    status.add_argument("job_id", type=uuid.UUID)
    return parser


def _job_json(job: Any) -> dict[str, Any]:
    return {
        "job_id": str(job.id),
        "course_session_id": str(job.course_session_id),
        "status": job.status,
        "stage": job.stage,
        "progress": job.progress,
        "duration_seconds": job.duration_seconds,
        "error_message": job.error_message,
    }


def _print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False))


async def cmd_enqueue(args: argparse.Namespace, settings: Settings) -> int:
    engine, session_factory = create_engine(settings)
    try:
        request = PodcastGenerationRequest(style=args.style, target_minutes=args.minutes, force=args.force)
        job, created = await enqueue_podcast_job(session_factory, args.session_id, request, settings)
    except CourseSessionNotFoundError:
        print(f"Session introuvable : {args.session_id}", file=sys.stderr)
        return 2
    finally:
        await engine.dispose()
    _print({**_job_json(job), "created": created})
    return 0


async def cmd_status(args: argparse.Namespace, settings: Settings) -> int:
    engine, session_factory = create_engine(settings)
    try:
        async with session_factory() as db:
            job = await jobs_repo.get_by_id(db, args.job_id)
    finally:
        await engine.dispose()
    if job is None:
        print(f"Job introuvable : {args.job_id}", file=sys.stderr)
        return 2
    _print(_job_json(job))
    return 0


async def cmd_run(args: argparse.Namespace, settings: Settings) -> int:
    engine, session_factory = create_engine(settings)
    tts_engine = PiperHTTPEngine.from_settings(settings)
    try:
        outcome = await run_podcast_job(
            args.job_id, session_factory=session_factory, gemini_client=GeminiClient(settings),
            tts_engine=tts_engine, settings=settings,
        )
        async with session_factory() as db:
            job = await jobs_repo.get_by_id(db, args.job_id)
    except JobNotFoundError:
        print(f"Job introuvable : {args.job_id}", file=sys.stderr)
        return 2
    finally:
        await tts_engine.close()
        await engine.dispose()
    _print({"outcome": outcome, **(_job_json(job) if job else {})})
    return 0 if outcome == "done" else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    handlers = {"enqueue": cmd_enqueue, "run": cmd_run, "status": cmd_status}
    return asyncio.run(handlers[args.command](args, get_settings()))


if __name__ == "__main__":
    sys.exit(main())
