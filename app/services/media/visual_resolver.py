"""Résolution des blocs IMAGE d'un cours : intention Gemini → `ApiImage` ré-hébergée.

Gemini n'émet jamais d'URL ni d'identifiant externe, seulement une intention (`image_source` +
`image_query`/`image_reference`) portée par des champs internes de `ApiContentBlock` (exclus de la
sérialisation API — voir app/api/schemas.py). Ce module la résout en une `MediaAsset` réellement
téléchargée/ré-encodée et stockée, servie ensuite par `GET /media/{asset_id}`.

Best-effort, ne lève jamais : un bloc dont la résolution échoue ou dépasse le budget de temps est
simplement retiré de la section (jamais de bloc IMAGE sans image, jamais de hotlink). Un bloc IMAGE
ne compte jamais dans la règle « visuel d'abord » (app/services/visual_validation.py), précisément
parce que sa résolution peut échouer.

Lot 1 (fondations) : aucun résolveur par source n'est encore branché, tout bloc IMAGE est donc
retiré (loggé `image_unresolved`). Les lots suivants brancheront `_resolve_pdf`, `_resolve_web`,
`_resolve_generated` sans changer ce squelette.
"""

from __future__ import annotations

import asyncio
import logging

from sqlalchemy.ext.asyncio import async_sessionmaker

from app.api.schemas import ApiContentBlock, ApiImage, ApiImageAttribution, CourseGenerationResponse, CourseSection
from app.core.config import Settings
from app.db.models import MediaAsset
from app.repositories import media_asset_repository
from app.services.media.storage import NormalizedImage, UnsupportedImageError, storage_path_for, write_to_disk

logger = logging.getLogger(__name__)


class ImageResolution:
    """Résultat d'un résolveur par source : image normalisée + métadonnées d'attribution."""

    __slots__ = ("image", "alt_text", "origin_url", "author", "license", "license_url", "source_filename", "source_page")

    def __init__(
        self,
        image: NormalizedImage,
        *,
        alt_text: str = "",
        origin_url: str | None = None,
        author: str | None = None,
        license: str | None = None,
        license_url: str | None = None,
        source_filename: str | None = None,
        source_page: int | None = None,
    ) -> None:
        self.image = image
        self.alt_text = alt_text
        self.origin_url = origin_url
        self.author = author
        self.license = license
        self.license_url = license_url
        self.source_filename = source_filename
        self.source_page = source_page


async def _resolve_pdf(block: ApiContentBlock, *, settings: Settings) -> tuple[ImageResolution, str] | None:
    """Figure extraite d'un PDF source (Lot 2). Non branché en Lot 1."""
    return None


async def _resolve_web(block: ApiContentBlock, *, settings: Settings) -> tuple[ImageResolution, str] | None:
    """Image web (Wikimedia Commons puis Openverse, Lot 3). Non branché en Lot 1."""
    return None


async def _resolve_generated(block: ApiContentBlock, *, settings: Settings) -> tuple[ImageResolution, str] | None:
    """Image générée par IA (Lot 5, sous flag). Non branché en Lot 1."""
    return None


_RESOLVERS = {
    "pdf": _resolve_pdf,
    "web": _resolve_web,
    "generated": _resolve_generated,
}


async def _resolve_block(
    block: ApiContentBlock,
    *,
    settings: Settings,
    session_factory: async_sessionmaker,
    semaphore: asyncio.Semaphore,
) -> ApiImage | None:
    resolver = _RESOLVERS.get(block.image_source or "")
    if resolver is None:
        logger.info("image_unresolved", extra={"reason": "unknown_source", "source": block.image_source})
        return None

    async with semaphore:
        try:
            result = await asyncio.wait_for(
                resolver(block, settings=settings), timeout=settings.media_resolve_timeout_seconds
            )
        except TimeoutError:
            logger.info("image_unresolved", extra={"reason": "timeout", "source": block.image_source})
            return None
        except UnsupportedImageError as exc:
            logger.info("image_unresolved", extra={"reason": "unsupported", "source": block.image_source, "detail": str(exc)})
            return None
        except Exception:
            logger.exception("image_resolve_failed", extra={"source": block.image_source})
            return None

    if result is None:
        logger.info("image_unresolved", extra={"reason": "no_candidate", "source": block.image_source})
        return None

    resolution, kind = result
    write_to_disk(settings, resolution.image)

    async with session_factory() as db:
        asset = await media_asset_repository.get_or_create(
            db,
            sha256=resolution.image.sha256,
            kind=kind,
            mime=resolution.image.mime,
            width=resolution.image.width,
            height=resolution.image.height,
            size_bytes=len(resolution.image.content),
            storage_path=str(storage_path_for(settings, resolution.image.sha256)),
            alt_text=resolution.alt_text or block.image_alt or "",
            origin_url=resolution.origin_url,
            author=resolution.author,
            license=resolution.license,
            license_url=resolution.license_url,
            source_filename=resolution.source_filename,
            source_page=resolution.source_page,
        )

    return _to_api_image(asset)


def _to_api_image(asset: MediaAsset) -> ApiImage:
    attribution = None
    if asset.author or asset.license:
        attribution = ApiImageAttribution(author=asset.author, license=asset.license, license_url=asset.license_url)
    return ApiImage(
        asset_id=str(asset.id),
        url=f"/media/{asset.id}",
        alt=asset.alt_text,
        caption="",
        width=asset.width,
        height=asset.height,
        attribution=attribution,
    )


async def resolve_visuals_in_sections(
    sections: list[CourseSection],
    *,
    settings: Settings,
    db_session_factory: async_sessionmaker | None = None,
) -> list[CourseSection]:
    """Résout tous les blocs IMAGE de `sections` (→ sous-sections). Best-effort, ne lève jamais.

    Portée volontairement limitée aux blocs typés (`subsections[].blocks[]`, voir `_map_subsections`
    dans course_generator.py) — utilisé aussi bien pour un cours complet que pour une seule section
    régénérée (`section_regenerator.py`).
    """
    if not sections:
        return sections
    if db_session_factory is None:
        for section in sections:
            for subsection in section.subsections:
                subsection.blocks = [b for b in subsection.blocks if b.type != "image"]
        return sections

    semaphore = asyncio.Semaphore(max(1, settings.media_resolve_concurrency))
    image_blocks = [
        block
        for section in sections
        for subsection in section.subsections
        for block in subsection.blocks
        if block.type == "image" and block.image_source
    ]

    resolved: dict[int, ApiImage] = {}
    if image_blocks:
        results = await asyncio.gather(
            *(
                _resolve_block(block, settings=settings, session_factory=db_session_factory, semaphore=semaphore)
                for block in image_blocks
            ),
            return_exceptions=False,
        )
        for block, image in zip(image_blocks, results, strict=True):
            if image is not None:
                resolved[id(block)] = image

    for section in sections:
        for subsection in section.subsections:
            kept: list[ApiContentBlock] = []
            for block in subsection.blocks:
                if block.type != "image":
                    kept.append(block)
                    continue
                image = resolved.get(id(block))
                if image is None:
                    continue  # non résolu : bloc retiré, jamais de bloc IMAGE sans image
                block.image = image
                if not block.image_caption:
                    block.image_caption = block.image_alt or ""
                kept.append(block)
            subsection.blocks = kept

    return sections


async def resolve_visuals(
    response: CourseGenerationResponse,
    *,
    settings: Settings,
    db_session_factory: async_sessionmaker | None = None,
) -> CourseGenerationResponse:
    """Résout tous les blocs IMAGE du cours. `response.answer` (réponse directe) reste hors champ : il
    n'est pas structuré par blocs typés."""
    if response.sections:
        await resolve_visuals_in_sections(response.sections, settings=settings, db_session_factory=db_session_factory)
    return response
