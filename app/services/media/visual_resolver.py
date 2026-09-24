"""Résolution des blocs IMAGE d'un cours : intention Gemini → `ApiImage` ré-hébergée.

Gemini n'émet jamais d'URL ni d'identifiant externe, seulement une intention (`image_source` +
`image_query`/`image_reference`) portée par des champs internes de `ApiContentBlock` (exclus de la
sérialisation API — voir app/api/schemas.py). Ce module la résout en une `MediaAsset` réellement
téléchargée/ré-encodée et stockée, servie ensuite par `GET /media/{asset_id}`.

Best-effort, ne lève jamais : un bloc dont la résolution échoue ou dépasse le budget de temps est
simplement retiré de la section (jamais de bloc IMAGE sans image, jamais de hotlink). Un bloc IMAGE
ne compte jamais dans la règle « visuel d'abord » (app/services/visual_validation.py), précisément
parce que sa résolution peut échouer.

Lot 1 (fondations) : squelette (sémaphore, timeout, dédup sha256, route GET /media/{id}).
Lot 3 : `_resolve_web` branché (Wikimedia Commons → Openverse, licence, vérification Gemini, cache).
`_resolve_pdf` (Lot 2) et `_resolve_generated` (Lot 5) restent non branchés.
"""

from __future__ import annotations

import asyncio
import logging

import httpx
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.api.schemas import ApiContentBlock, ApiImage, ApiImageAttribution, CourseGenerationResponse, CourseSection
from app.core.config import Settings
from app.core.exceptions import GeminiServiceError, MediaFetchRejected, MediaFetchTooLarge, MediaWebError
from app.db.models import MediaAsset
from app.repositories import media_asset_repository, media_query_cache_repository
from app.services.gemini_client import GeminiClient
from app.services.media import licenses
from app.services.media.fetcher import fetch_image
from app.services.media.providers import ImageProvider, WebImageCandidate
from app.services.media.providers.openverse import OpenverseProvider
from app.services.media.providers.wikimedia import WikimediaProvider
from app.services.media.storage import NormalizedImage, UnsupportedImageError, normalize_image, storage_path_for, write_to_disk
from app.services.media.web_ranker import rank_web_images

logger = logging.getLogger(__name__)


class ImageResolution:
    """Résultat d'un résolveur par source : image normalisée + métadonnées d'attribution."""

    __slots__ = (
        "image", "alt_text", "origin_url", "author", "license", "license_url",
        "source_filename", "source_page", "cache_key",
    )

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
        cache_key: str | None = None,
    ) -> None:
        self.image = image
        self.alt_text = alt_text
        self.origin_url = origin_url
        self.author = author
        self.license = license
        self.license_url = license_url
        self.source_filename = source_filename
        self.source_page = source_page
        # Clé du cache de requête (media_query_cache) à écrire une fois l'asset créé — voir
        # _resolve_block, qui seul connaît l'asset_id final (dédup sha256 comprise).
        self.cache_key = cache_key


async def _resolve_pdf(
    block: ApiContentBlock, *, settings: Settings, gemini_client: GeminiClient | None,
    session_factory: async_sessionmaker | None, section_title: str,
) -> ImageResolution | MediaAsset | None:
    """Figure extraite d'un PDF source (Lot 2). Non branché."""
    return None


def _build_web_providers(settings: Settings) -> list[ImageProvider]:
    providers: list[ImageProvider] = []
    for name in settings.media_web_providers:
        if name == "wikimedia":
            providers.append(
                WikimediaProvider(user_agent=settings.media_web_user_agent, timeout_seconds=settings.media_web_timeout_seconds)
            )
        elif name == "openverse":
            providers.append(
                OpenverseProvider(
                    user_agent=settings.media_web_user_agent,
                    client_id=settings.openverse_client_id,
                    client_secret=settings.openverse_client_secret.get_secret_value() if settings.openverse_client_secret else None,
                    timeout_seconds=settings.media_web_timeout_seconds,
                )
            )
        else:
            logger.warning("image_web_unknown_provider", extra={"provider": name})
    return providers


async def _search_web_candidates(query: str, *, settings: Settings) -> list[WebImageCandidate]:
    """Commons d'abord, puis Openverse seulement si Commons n'a pas suffi. Filtre licence et taille."""
    collected: list[WebImageCandidate] = []
    for provider in _build_web_providers(settings):
        try:
            if len(collected) < settings.media_web_max_candidates:
                found = await provider.search(query, limit=settings.media_web_max_candidates)
                collected.extend(found)
        except MediaWebError as exc:
            logger.warning("image_web_provider_failed", extra={"provider": provider.name, "error": str(exc)})
        finally:
            aclose = getattr(provider, "aclose", None)
            if aclose is not None:
                await aclose()

    filtered: list[WebImageCandidate] = []
    for candidate in collected:
        code = licenses.normalize_license(candidate.license)
        if not licenses.is_allowed(code, settings):
            logger.info("image_web_rejected", extra={"reason": "license", "raw_license": candidate.license})
            continue
        if candidate.width and candidate.width < settings.media_web_min_width:
            logger.info("image_web_rejected", extra={"reason": "too_small", "width": candidate.width})
            continue
        filtered.append(candidate)
    return filtered[: settings.media_web_max_candidates]


async def _download_and_normalize(
    candidates: list[WebImageCandidate], *, settings: Settings
) -> list[tuple[NormalizedImage, WebImageCandidate]]:
    """Télécharge et normalise chaque candidat ; ceux qui échouent (fetch ou format) sont ignorés,
    jamais fatals pour les autres."""
    results: list[tuple[NormalizedImage, WebImageCandidate]] = []
    async with httpx.AsyncClient(
        timeout=settings.media_web_timeout_seconds, headers={"User-Agent": settings.media_web_user_agent},
        follow_redirects=False,
    ) as client:
        for candidate in candidates:
            try:
                raw = await fetch_image(candidate.download_url, settings=settings, client=client)
                image = normalize_image(raw, settings=settings)
            except (MediaFetchRejected, MediaFetchTooLarge, UnsupportedImageError) as exc:
                logger.info("image_web_rejected", extra={"reason": "fetch_or_normalize", "detail": str(exc)})
                continue
            results.append((image, candidate))
    return results


async def _cache_negative(session_factory: async_sessionmaker, query_hash: str, query: str) -> None:
    async with session_factory() as db:
        await media_query_cache_repository.upsert(db, query_hash, query, None)


async def _resolve_web(
    block: ApiContentBlock, *, settings: Settings, gemini_client: GeminiClient | None,
    session_factory: async_sessionmaker | None, section_title: str,
) -> ImageResolution | MediaAsset | None:
    """Image web (Wikimedia Commons puis Openverse, Lot 3)."""
    if not settings.media_web_enabled or not settings.media_web_user_agent:
        logger.info("image_web_disabled", extra={"reason": "config"})
        return None
    query = (block.image_query or "").strip()
    if not query or session_factory is None:
        return None

    query_hash = media_query_cache_repository.normalize_query_key(query)
    async with session_factory() as db:
        hit = await media_query_cache_repository.get_fresh(db, query_hash, settings.media_web_cache_ttl)
    if hit.found:
        if hit.asset_id is None:
            logger.info("image_web_cache_hit", extra={"result": "negative"})
            return None
        async with session_factory() as db:
            asset = await media_asset_repository.get(db, hit.asset_id)
        if asset is not None:
            logger.info("image_web_cache_hit", extra={"result": "positive"})
            return asset
        # L'asset a disparu depuis (jamais censé arriver en usage normal) : recherche fraîche.

    candidates = await _search_web_candidates(query, settings=settings)
    logger.info("image_web_search", extra={"query": query, "candidates": len(candidates)})
    if not candidates:
        await _cache_negative(session_factory, query_hash, query)
        return None

    normalized = await _download_and_normalize(candidates, settings=settings)
    if not normalized:
        await _cache_negative(session_factory, query_hash, query)
        return None

    chosen_index = 0
    if gemini_client is not None:
        try:
            chosen_index = await rank_web_images(
                [(img, cand.title) for img, cand in normalized],
                image_alt=block.image_alt or "", image_query=query, section_title=section_title,
                gemini_client=gemini_client, settings=settings,
            )
        except GeminiServiceError as exc:
            logger.warning("image_verify_failed", extra={"error": str(exc)})
            chosen_index = 0
    if chosen_index is None:
        await _cache_negative(session_factory, query_hash, query)
        return None

    image, candidate = normalized[chosen_index]
    license_code = licenses.normalize_license(candidate.license)
    if not licenses.is_allowed(license_code, settings):
        # Défense en profondeur : déjà filtré dans _search_web_candidates, mais jamais retenu sans
        # licence autorisée même si ce filtre venait à changer.
        logger.info("image_web_rejected", extra={"reason": "license", "raw_license": candidate.license})
        await _cache_negative(session_factory, query_hash, query)
        return None

    return ImageResolution(
        image,
        alt_text=block.image_alt or candidate.title,
        origin_url=candidate.page_url,
        author=licenses.strip_html(candidate.author),
        license=license_code,
        license_url=candidate.license_url or licenses.canonical_url(license_code),
        cache_key=query_hash,
    )


async def _resolve_generated(
    block: ApiContentBlock, *, settings: Settings, gemini_client: GeminiClient | None,
    session_factory: async_sessionmaker | None, section_title: str,
) -> ImageResolution | MediaAsset | None:
    """Image générée par IA (Lot 5, sous flag). Non branché."""
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
    gemini_client: GeminiClient | None,
    section_title: str,
) -> ApiImage | None:
    resolver = _RESOLVERS.get(block.image_source or "")
    if resolver is None:
        logger.info("image_unresolved", extra={"reason": "unknown_source", "source": block.image_source})
        return None

    async with semaphore:
        try:
            result = await asyncio.wait_for(
                resolver(
                    block, settings=settings, gemini_client=gemini_client,
                    session_factory=session_factory, section_title=section_title,
                ),
                timeout=settings.media_resolve_timeout_seconds,
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

    if isinstance(result, MediaAsset):
        return _to_api_image(result)

    resolution = result
    write_to_disk(settings, resolution.image)

    async with session_factory() as db:
        asset = await media_asset_repository.get_or_create(
            db,
            sha256=resolution.image.sha256,
            kind=block.image_source,
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
        if resolution.cache_key:
            await media_query_cache_repository.upsert(db, resolution.cache_key, block.image_query or "", asset.id)

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
    gemini_client: GeminiClient | None = None,
) -> list[CourseSection]:
    """Résout tous les blocs IMAGE de `sections` (→ sous-sections). Best-effort, ne lève jamais.

    Portée volontairement limitée aux blocs typés (`subsections[].blocks[]`, voir `_map_subsections`
    dans course_generator.py) — utilisé aussi bien pour un cours complet que pour une seule section
    régénérée (`section_regenerator.py`). `gemini_client` sert à la vérification de pertinence des
    images web (Lot 3) : sans lui, `_resolve_web` retient le premier candidat filtré.
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
        (block, section.title)
        for section in sections
        for subsection in section.subsections
        for block in subsection.blocks
        if block.type == "image" and block.image_source
    ]

    resolved: dict[int, ApiImage] = {}
    if image_blocks:
        results = await asyncio.gather(
            *(
                _resolve_block(
                    block, settings=settings, session_factory=db_session_factory, semaphore=semaphore,
                    gemini_client=gemini_client, section_title=section_title,
                )
                for block, section_title in image_blocks
            ),
            return_exceptions=False,
        )
        for (block, _title), image in zip(image_blocks, results, strict=True):
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
    gemini_client: GeminiClient | None = None,
) -> CourseGenerationResponse:
    """Résout tous les blocs IMAGE du cours. `response.answer` (réponse directe) reste hors champ : il
    n'est pas structuré par blocs typés."""
    if response.sections:
        await resolve_visuals_in_sections(
            response.sections, settings=settings, db_session_factory=db_session_factory, gemini_client=gemini_client,
        )
    return response
