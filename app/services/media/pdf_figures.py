"""Figures des PDF ingérés : stockage à l'ingestion, pour les blocs IMAGE `image_source="pdf"`.

Chaque image jugée informative par Gemini (`app.services.gemini_vision`) est ré-encodée et stockée
comme n'importe quelle autre image (`media_assets`, jamais l'octet brut du PDF), puis reliée à son
fichier par une référence courte `fig-…`. Cette référence est inscrite dans le chunk de description
de la figure : le modèle ne peut citer qu'une figure présente dans le contexte qu'il a reçu, et la
référence, dérivée du contenu de l'image, ne se devine pas.

Best-effort par image : une figure non stockable (format non supporté, base indisponible) reste
décrite en texte comme avant, sans jamais faire échouer l'ingestion du PDF.
"""

from __future__ import annotations

import asyncio
import logging
import re

from sqlalchemy.ext.asyncio import async_sessionmaker

from app.core.config import Settings
from app.repositories import ingested_file_figure_repository, media_asset_repository
from app.services.gemini_vision import KeyImage
from app.services.media.storage import NormalizedImage, UnsupportedImageError, normalize_image, storage_path_for, write_to_disk

logger = logging.getLogger(__name__)

_REF_HEX_LENGTH = 12
FIGURE_REF_RE = re.compile(rf"fig-[0-9a-f]{{{_REF_HEX_LENGTH}}}")
# Texte alternatif de l'image : la description complète reste dans le chunk indexé.
_ALT_TEXT_MAX = 300


def figure_ref(sha256: str) -> str:
    """Référence d'une figure, dérivée du contenu ré-encodé (stable d'une ingestion à l'autre)."""
    return f"fig-{sha256[:_REF_HEX_LENGTH]}"


def _normalize_and_write(image: KeyImage, settings: Settings) -> NormalizedImage:
    normalized = normalize_image(image.data, settings=settings)
    write_to_disk(settings, normalized)
    return normalized


async def _store_one(
    image: KeyImage, filename: str, *, settings: Settings, session_factory: async_sessionmaker
) -> str | None:
    try:
        # Pillow et l'écriture disque sont synchrones : hors de la boucle événementielle.
        normalized = await asyncio.to_thread(_normalize_and_write, image, settings)
    except UnsupportedImageError as exc:
        logger.info("pdf_figure_skipped", extra={"pdf_filename": filename, "page": image.page, "detail": str(exc)})
        return None

    ref = figure_ref(normalized.sha256)
    try:
        async with session_factory() as db:
            asset = await media_asset_repository.get_or_create(
                db,
                sha256=normalized.sha256,
                kind="source",
                mime=normalized.mime,
                width=normalized.width,
                height=normalized.height,
                size_bytes=len(normalized.content),
                storage_path=str(storage_path_for(settings, normalized.sha256)),
                alt_text=image.description[:_ALT_TEXT_MAX],
                source_filename=filename,
                source_page=image.page,
            )
            await ingested_file_figure_repository.add(
                db, filename=filename, ref=ref, page=image.page, asset_id=asset.id
            )
    except Exception:
        logger.warning("pdf_figure_store_failed", extra={"pdf_filename": filename, "page": image.page}, exc_info=True)
        return None
    return ref


async def store_pdf_figures(
    images: list[KeyImage], filename: str, *, settings: Settings, session_factory: async_sessionmaker | None
) -> list[str | None]:
    """Stocke les figures d'un PDF ; une référence par image (même ordre), None si non stockée.

    Sans `session_factory` (pas de base), aucune figure n'est stockée.
    """
    if session_factory is None:
        return [None] * len(images)
    return [
        await _store_one(image, filename, settings=settings, session_factory=session_factory) for image in images
    ]
