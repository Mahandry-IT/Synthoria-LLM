"""Video generation orchestration service.

Translates a course session's gemini_response into a storyboard,
generates video clips via HF Inference, concatenates them with ffmpeg,
and persists the result.

Executed as a FastAPI BackgroundTask — not a standalone worker.
"""

import json
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.core.exceptions import (
    HFVideoGenerationError,
    HFVideoServiceError,
    HFVideoUnavailableError,
)
from app.repositories import video_job_repository
from app.schemas.video_generation import VideoJobStatus

logger = logging.getLogger(__name__)

_MAX_PROMPT_CHARS = 2000  # Limite la taille du prompt envoyé à HF


@dataclass
class Scene:
    """Scène de storyboard pour la génération vidéo."""

    narration: str
    visual_prompt: str
    duration_seconds: int


def _load_video_instructions() -> str:
    """Charge les instructions de style vidéo tutoriel."""
    candidates = [
        Path(__file__).resolve().parents[2] / "instruction" / "video_tutorial_instructions.md",
        Path(__file__).resolve().parents[1] / "instruction" / "video_tutorial_instructions.md",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    logger.warning("video_tutorial_instructions_missing")
    return ""


def build_video_prompt_from_course(gemini_response: dict[str, Any]) -> list[Scene]:
    """Transforme le JSON du cours en storyboard de scènes vidéo.

    Extraction déterministe V1 (pas d'appel Gemini) :
    - Titre du cours → scène d'accroche
    - Sections développement → scènes pédagogiques
    - Pièges fréquents → scène pièges
    - Résumé → scène conclusion

    Le prompt visuel est construit à partir des titres et du contenu
    textuel des sections, tronqué pour respecter les limites du modèle.
    """
    scenes: list[Scene] = []
    meta = gemini_response.get("meta", {})
    title = meta.get("title", "cours")
    subject = meta.get("subject", "")

    # Scène d'accroche
    scenes.append(Scene(
        narration=f"Bienvenue dans ce cours sur {title}. Aujourd'hui, on va voir ensemble {subject}.",
        visual_prompt=f"A bright, clean whiteboard. The title '{title}' appears in bold blue marker, written by an invisible hand. Camera zooms in slowly.",
        duration_seconds=6,
    ))

    # Sections de développement
    sections = gemini_response.get("sections", [])
    for section in sections:
        section_title = section.get("title", "")
        quoi = ""
        pourquoi = ""
        comment = ""
        # Extraire depuis les subsections ou directement
        if isinstance(section, dict):
            quoi = section.get("quoi", "")
            pourquoi = section.get("pourquoi", "")
            comment = section.get("comment", "")
            # Fallback sur subsections (format Gemini brut)
            if not quoi:
                for sub in section.get("subsections", []):
                    sub_title = sub.get("title", "").lower()
                    sub_text = " ".join(
                        b.get("text", "")
                        for b in sub.get("blocks", [])
                        if b.get("text")
                    )
                    if "quoi" in sub_title:
                        quoi = sub_text
                    elif "pourquoi" in sub_title:
                        pourquoi = sub_text
                    elif "comment" in sub_title:
                        comment = sub_text

        if not section_title:
            continue

        # Scène "Quoi" — définition
        if quoi:
            narration_text = f"Commençons par {section_title}. {quoi[:150]}"
            visual_text = (
                f"Whiteboard with the heading '{section_title[:60]}' at the top. "
                f"A diagram or key concept appears below it, drawn in real-time with blue and black markers. "
                f"Camera is steady, medium shot."
            )
            scenes.append(Scene(narration=narration_text, visual_prompt=visual_text, duration_seconds=8))

        # Scène "Pourquoi" — importance
        if pourquoi:
            narration_text = f"Pourquoi est-ce important ? {pourquoi[:150]}"
            visual_text = (
                f"Camera zooms into a highlighted section of the whiteboard. "
                f"A glowing arrow points to the key formula or concept. "
                f"Colors shift to emphasize importance — orange highlights appear."
            )
            scenes.append(Scene(narration=narration_text, visual_prompt=visual_text, duration_seconds=7))

        # Scène "Comment" — mécanisme + exemple
        if comment:
            narration_text = f"Comment ça marche concrètement ? {comment[:200]}"
            visual_text = (
                f"Close-up on the whiteboard. A hand writes step-by-step calculations "
                f"with numbers appearing one by one. Each step is circled or underlined "
                f"as it is completed. Camera follows the writing from left to right."
            )
            scenes.append(Scene(narration=narration_text, visual_prompt=visual_text, duration_seconds=10))

    # Pièges fréquents
    pitfalls = gemini_response.get("common_pitfalls", [])
    if pitfalls:
        pitfall = pitfalls[0]  # Premier piège le plus important
        desc = pitfall.get("description", "") if isinstance(pitfall, dict) else ""
        avoid = pitfall.get("how_to_avoid", "") if isinstance(pitfall, dict) else ""
        if desc:
            scenes.append(Scene(
                narration=f"Attention au piège : {desc[:150]}. {avoid[:100]}",
                visual_prompt=(
                    "Split screen on whiteboard: left side shows a large red X over a wrong formula, "
                    "right side shows a green checkmark over the correct version. "
                    "Camera zooms into the correct version."
                ),
                duration_seconds=8,
            ))

    # Scène conclusion
    scenes.append(Scene(
        narration=f"En résumé, {title} repose sur les points clés qu'on vient de voir. N'hésite pas à revoir les étapes si nécessaire.",
        visual_prompt=(
            "Camera pulls back to show the full whiteboard with all key concepts visible. "
            "A summary list appears as bullet points, each one highlighting briefly. "
            "Fade to a clean end screen."
        ),
        duration_seconds=7,
    ))

    return scenes


def _build_scenes_text(scenes: list[Scene]) -> str:
    """Sérialise les scènes en texte pour le prompt HF."""
    parts = []
    for i, scene in enumerate(scenes, 1):
        parts.append(
            f"Scene {i} ({scene.duration_seconds}s):\n"
            f"  Narration: {scene.narration}\n"
            f"  Visual: {scene.visual_prompt}"
        )
    return "\n\n".join(parts)


def _concatenate_clips(clips: list[Path], output: Path) -> None:
    """Concatène des clips vidéo MP4 via ffmpeg.

    Utilise un fichier de liste pour éviter les problèmes d'escaping.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, prefix="ffmpeg_list_"
    ) as list_file:
        for clip in clips:
            list_file.write(f"file '{clip}'\n")
        list_path = list_file.name

    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", list_path,
                "-c", "copy",
                str(output),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr[:500]}")
    finally:
        Path(list_path).unlink(missing_ok=True)


async def generate_course_video(
    course_session_id: str,
    gemini_response: dict[str, Any],
    session_factory: Any,
) -> None:
    """Point d'entrée exécuté comme BackgroundTask.

    1. Crée le job en DB (pending)
    2. Build le storyboard depuis gemini_response
    3. Génère clip par clip via HF
    4. Concatène avec ffmpeg
    5. Met à jour le job (succeeded / failed)
    """
    from uuid import UUID

    from sqlalchemy.ext.asyncio import async_sessionmaker

    from app.db.models import CourseSession

    sf: async_sessionmaker = session_factory
    job = None

    async with sf() as session:
        # Récupérer la session de cours
        cs = await session.get(CourseSession, UUID(course_session_id))
        if cs is None:
            logger.error("video_gen_session_not_found", extra={"session_id": course_session_id})
            return

        # Créer le job
        job = await video_job_repository.create(session, course_session_id=cs.id)
        logger.info("video_job_created", extra={"job_id": str(job.id)})

        # Marquer en cours
        await video_job_repository.update_status(session, job.id, status="running")

    # Import différé pour éviter les circular imports
    from app.services.hf_video_client import HFVideoClient

    settings = Settings()
    hf_client = HFVideoClient(settings)
    storage_path = Path(settings.video_storage_path)
    storage_path.mkdir(parents=True, exist_ok=True)

    clips_dir = Path(tempfile.mkdtemp(prefix="video_clips_"))

    try:
        # Build storyboard
        scenes = build_video_prompt_from_course(gemini_response)
        if not scenes:
            async with sf() as session:
                await video_job_repository.update_status(
                    session, job.id,
                    status="failed",
                    error_message="Aucune scène générée à partir du cours",
                )
            return

        logger.info("video_storyboard_built", extra={"job_id": str(job.id), "scene_count": len(scenes)})

        # Générer chaque clip
        clip_paths: list[Path] = []
        for i, scene in enumerate(scenes):
            # Tronquer le prompt visuel
            prompt = scene.visual_prompt[:_MAX_PROMPT_CHARS]
            try:
                video_bytes, model_used, fallback_used = await hf_client.generate_video_with_fallback(prompt)
                clip_path = clips_dir / f"clip_{i:03d}.mp4"
                clip_path.write_bytes(video_bytes)
                clip_paths.append(clip_path)
                logger.info(
                    "video_clip_generated",
                    extra={"job_id": str(job.id), "clip": i, "model": model_used},
                )
            except (HFVideoServiceError, HFVideoUnavailableError) as exc:
                logger.error(
                    "video_clip_failed",
                    extra={"job_id": str(job.id), "clip": i, "error": str(exc)},
                )
                # Mettre à jour le job avec l'erreur
                async with sf() as session:
                    await video_job_repository.update_status(
                        session, job.id,
                        status="failed",
                        model_used=settings.hf_video_model_primary,
                        fallback_used=fallback_used if 'fallback_used' in dir() else False,
                        error_message=f"Clip {i} échoué: {exc}",
                    )
                return

        # Concaténer les clips
        if len(clip_paths) == 1:
            final_path = storage_path / f"{job.id}.mp4"
            shutil.copy2(clip_paths[0], final_path)
        else:
            final_path = storage_path / f"{job.id}.mp4"
            _concatenate_clips(clip_paths, final_path)

        # Mettre à jour le job en succès
        async with sf() as session:
            await video_job_repository.update_status(
                session, job.id,
                status="succeeded",
                model_used=settings.hf_video_model_primary,
                video_path=str(final_path),
            )
        logger.info("video_generation_succeeded", extra={"job_id": str(job.id), "path": str(final_path)})

    except Exception as exc:
        logger.error("video_generation_failed", extra={"job_id": str(job.id) if job else None, "error": str(exc)})
        if job:
            async with sf() as session:
                await video_job_repository.update_status(
                    session, job.id,
                    status="failed",
                    error_message=str(exc),
                )
    finally:
        # Nettoyage des clips intermédiaires
        shutil.rmtree(clips_dir, ignore_errors=True)
