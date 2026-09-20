"""Routes du podcast : création de job (202 + polling), état, audio, transcript, script."""

import logging
import re
import time
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Body, Depends, HTTPException, Request, status
from fastapi.responses import FileResponse, Response, StreamingResponse
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.api.schemas import (
    PodcastGenerationRequest,
    PodcastJobList,
    PodcastJobResponse,
    PodcastJobStatus,
)
from app.core.config import Settings, get_settings
from app.repositories import course_session_repository, podcast_job_repository
from app.schemas.podcast import PodcastScript
from app.services.podcast.course_serializer import has_usable_content, serialize_course
from app.services.podcast.jobs import CourseSessionNotFoundError, enqueue_podcast_job
from app.services.podcast.pipeline import audio_file, transcript_file

logger = logging.getLogger(__name__)

router = APIRouter(tags=["podcasts"])

_RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")
_CHUNK = 64 * 1024


class _RateLimiter:
    """Fenêtre glissante par IP (mémoire) — limite dédiée à la création de jobs, coûteuse."""

    def __init__(self) -> None:
        self._hits: dict[str, list[float]] = defaultdict(list)

    def check(self, key: str, limit: int) -> None:
        now = time.monotonic()
        recent = [t for t in self._hits[key] if now - t < 60]
        if len(recent) >= limit:
            self._hits[key] = recent
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Trop de demandes de podcast, réessayez plus tard",
            )
        recent.append(now)
        self._hits[key] = recent


def _limit_generation(request: Request, settings: Settings = Depends(get_settings)) -> None:
    limiter = getattr(request.app.state, "podcast_rate_limiter", None)
    if limiter is None:
        limiter = request.app.state.podcast_rate_limiter = _RateLimiter()
    limiter.check(request.client.host if request.client else "unknown", settings.podcast_generate_rate_limit_per_minute)


def _require_enabled(settings: Settings = Depends(get_settings)) -> None:
    if not settings.podcast_enabled:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Génération de podcast désactivée")


def _session_factory(request: Request) -> async_sessionmaker:
    return request.app.state.db_session_factory


def _job_status(job) -> PodcastJobStatus:
    return PodcastJobStatus(
        job_id=job.id, course_session_id=job.course_session_id, status=job.status, stage=job.stage,
        progress=job.progress, error_message=job.error_message, duration_seconds=job.duration_seconds,
        created_at=job.created_at, updated_at=job.updated_at,
    )


async def _get_job_or_404(request: Request, job_id: UUID):
    async with _session_factory(request)() as db:
        job = await podcast_job_repository.get_by_id(db, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job de podcast introuvable")
    return job


def _require_done(job) -> None:
    if job.status != "done":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Le podcast n'est pas encore prêt")


def _parse_range(header: str, size: int) -> tuple[int, int] | None:
    """Plage d'octets (début, fin incluse) d'un en-tête Range à une seule plage ; None si invalide."""
    match = _RANGE_RE.match(header.strip())
    if not match:
        return None
    first, last = match.groups()
    if first == "" and last == "":
        return None
    if first == "":  # suffixe : les `last` derniers octets
        length = int(last)
        if length == 0:
            return None
        return max(size - length, 0), size - 1
    start = int(first)
    end = min(int(last), size - 1) if last else size - 1
    if start > end or start >= size:
        return None
    return start, end


def _iter_range(path: Path, start: int, end: int) -> Iterator[bytes]:
    remaining = end - start + 1
    with path.open("rb") as handle:
        handle.seek(start)
        while remaining > 0:
            chunk = handle.read(min(_CHUNK, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


def _file_response(request: Request, path: Path, media_type: str) -> Response:
    """Fichier avec support Range (seek du lecteur audio) — Starlette 0.38 ne le gère pas."""
    if not path.is_file():
        raise HTTPException(status_code=status.HTTP_410_GONE, detail="Fichier expiré ou supprimé")
    size = path.stat().st_size
    header = request.headers.get("range")
    if not header:
        return FileResponse(path, media_type=media_type, headers={"Accept-Ranges": "bytes"})
    byte_range = _parse_range(header, size)
    if byte_range is None:
        return Response(status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE, headers={"Content-Range": f"bytes */{size}"})
    start, end = byte_range
    return StreamingResponse(
        _iter_range(path, start, end),
        status_code=status.HTTP_206_PARTIAL_CONTENT,
        media_type=media_type,
        headers={
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Content-Length": str(end - start + 1),
            "Accept-Ranges": "bytes",
        },
    )


@router.post(
    "/podcasts/generate/{session_id}",
    response_model=PodcastJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(_require_enabled), Depends(_limit_generation)],
)
async def generate_podcast(
    request: Request,
    session_id: UUID,
    body: PodcastGenerationRequest | None = Body(None),
    settings: Settings = Depends(get_settings),
) -> PodcastJobResponse:
    """Met en file la génération d'un podcast (202) ; suivre l'avancement via GET /podcasts/jobs/{job_id}."""
    async with _session_factory(request)() as db:
        course_row = await course_session_repository.get_by_id(db, session_id)
    if course_row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session de cours introuvable")
    if not has_usable_content(serialize_course(course_row.gemini_response)):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Le cours ne contient aucun contenu exploitable"
        )
    try:
        job, _ = await enqueue_podcast_job(_session_factory(request), session_id, body, settings)
    except CourseSessionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session de cours introuvable") from exc
    return PodcastJobResponse(job_id=job.id, status=job.status)


@router.get("/podcasts/jobs/{job_id}", response_model=PodcastJobStatus)
async def get_podcast_job(request: Request, job_id: UUID) -> PodcastJobStatus:
    return _job_status(await _get_job_or_404(request, job_id))


@router.get("/podcasts/{job_id}/audio")
async def get_podcast_audio(
    request: Request, job_id: UUID, settings: Settings = Depends(get_settings)
) -> Response:
    job = await _get_job_or_404(request, job_id)
    _require_done(job)
    return _file_response(request, audio_file(settings, job_id), "audio/mpeg")


@router.get("/podcasts/{job_id}/transcript")
async def get_podcast_transcript(
    request: Request, job_id: UUID, settings: Settings = Depends(get_settings)
) -> Response:
    job = await _get_job_or_404(request, job_id)
    _require_done(job)
    return _file_response(request, transcript_file(settings, job_id), "text/vtt; charset=utf-8")


@router.get("/podcasts/{job_id}/script", response_model=PodcastScript)
async def get_podcast_script(request: Request, job_id: UUID) -> PodcastScript:
    job = await _get_job_or_404(request, job_id)
    if not job.script:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Le script n'est pas encore généré")
    return PodcastScript.model_validate(job.script)


@router.get("/courses/history/{session_id}/podcasts", response_model=PodcastJobList)
async def list_session_podcasts(request: Request, session_id: UUID) -> PodcastJobList:
    async with _session_factory(request)() as db:
        jobs = await podcast_job_repository.list_by_session(db, session_id)
    return PodcastJobList(data=[_job_status(j) for j in jobs])
