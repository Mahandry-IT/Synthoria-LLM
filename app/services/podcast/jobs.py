"""Création idempotente de jobs podcast et résolution sûre des chemins de stockage."""

import hashlib
import json
import uuid
from pathlib import Path

from sqlalchemy.ext.asyncio import async_sessionmaker

from app.api.schemas import PodcastGenerationRequest
from app.core.config import Settings
from app.db.models import PodcastJob
from app.repositories import course_session_repository, podcast_job_repository


class CourseSessionNotFoundError(Exception):
    """Aucune session de cours ne correspond à l'id fourni."""


def resolve_params(request: PodcastGenerationRequest | None, settings: Settings) -> dict[str, object]:
    """Paramètres effectifs du job (défauts serveur appliqués, durée plafonnée)."""
    request = request or PodcastGenerationRequest()
    minutes = request.target_minutes or settings.podcast_default_target_minutes
    return {"style": request.style, "target_minutes": min(minutes, settings.podcast_max_minutes)}


def compute_params_hash(params: dict[str, object]) -> str:
    """Hash stable (ordre des clés indifférent) des paramètres d'un job."""
    return hashlib.sha256(json.dumps(params, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def job_dir(settings: Settings, job_id: uuid.UUID) -> Path:
    """Dossier du job, dérivé du seul job_id côté serveur et vérifié contre le path traversal."""
    root = Path(settings.podcast_storage_dir).resolve()
    directory = (root / str(job_id)).resolve()
    if root != directory and root not in directory.parents:
        raise ValueError("Chemin de job hors du dossier de stockage")
    return directory


async def enqueue_podcast_job(
    session_factory: async_sessionmaker,
    course_session_id: uuid.UUID,
    request: PodcastGenerationRequest | None,
    settings: Settings,
) -> tuple[PodcastJob, bool]:
    """Crée un job, ou retourne l'existant non échoué pour les mêmes paramètres.

    Retour : (job, créé). request.force=True crée toujours un nouveau job.
    Lève CourseSessionNotFoundError si la session de cours n'existe pas.
    """
    params = resolve_params(request, settings)
    params_hash = compute_params_hash(params)
    force = bool(request and request.force)

    async with session_factory() as db:
        if await course_session_repository.get_by_id(db, course_session_id) is None:
            raise CourseSessionNotFoundError(str(course_session_id))
        if not force:
            existing = await podcast_job_repository.find_reusable(db, course_session_id, params_hash)
            if existing is not None:
                return existing, False
        job = await podcast_job_repository.create(
            db, course_session_id=course_session_id, params=params, params_hash=params_hash
        )
        return job, True
