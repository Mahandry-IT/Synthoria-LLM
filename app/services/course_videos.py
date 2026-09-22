"""Orchestration des vidéos d'un cours.

V1 : YouTube Data API v3 (recherche réelle à partir de `video_search_queries`, jamais d'ID
proposé par Gemini) en priorité ; repli sur la recherche groundée Gemini + vérification oEmbed
(comportement historique) si aucune clé n'est configurée, si le quota Data API est épuisé, ou si
aucun résultat exploitable n'a été trouvé. `attach_verified_videos` ne lève jamais.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from itertools import zip_longest
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy.ext.asyncio import async_sessionmaker

from app.api.schemas import CourseGenerationResponse, CourseVideo
from app.core.config import Settings
from app.core.exceptions import YoutubeQuotaExceeded, YoutubeServiceError
from app.repositories import youtube_search_cache_repository as cache_repo
from app.services.gemini_client import GeminiClient
from app.services.youtube import candidate_video, resolve_grounding_video_ids, verify_videos
from app.services.youtube_data_client import YoutubeDataClient, YoutubeVideoDetails

logger = logging.getLogger(__name__)

_YOUTUBE_URL_RE = re.compile(
    r"https?://(?:www\.|m\.)?(?:youtube\.com/(?:watch\?v=|embed/|shorts/)|youtu\.be/)[A-Za-z0-9_-]{11}"
)
_DURATION_RE = re.compile(r"^P(?:(?P<days>\d+)D)?T?(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?$")
_QUERY_MAX_CHARS = 100
_PACIFIC = ZoneInfo("America/Los_Angeles")


def parse_iso8601_duration(value: str) -> int | None:
    """Durée en secondes depuis une durée ISO 8601 (ex. `PT4M13S`, `PT1H2M`) ; None si invalide/vide."""
    if not value:
        return None
    match = _DURATION_RE.match(value)
    if not match or not any(match.groups()):
        return None
    parts = {k: int(v) if v else 0 for k, v in match.groupdict().items()}
    return parts["days"] * 86400 + parts["hours"] * 3600 + parts["minutes"] * 60 + parts["seconds"]


# ─── Circuit breaker quota (best-effort, par processus) ──────


class _QuotaBreaker:
    """Désactive la Data API jusqu'au prochain reset (minuit heure du Pacifique) après un 429/quota."""

    def __init__(self) -> None:
        self._open_until: datetime | None = None

    def trip(self) -> None:
        now = datetime.now(_PACIFIC)
        reset = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        self._open_until = reset.astimezone(timezone.utc)
        logger.warning("youtube_quota_breaker_open", extra={"reset_at": self._open_until.isoformat()})

    def is_open(self) -> bool:
        return self._open_until is not None and datetime.now(timezone.utc) < self._open_until


_breaker = _QuotaBreaker()


def reset_quota_breaker() -> None:
    """Réservé aux tests : force la fermeture du disjoncteur entre deux scénarios."""
    _breaker._open_until = None  # noqa: SLF001


# ─── Filtres et fusion, purs et testables ────────────────────


def passes_filters(details: YoutubeVideoDetails, settings: Settings) -> bool:
    """Vrai si la vidéo est intégrable, publique, pas un direct, et dans la fenêtre de durée voulue."""
    if not details.embeddable or details.privacy_status != "public" or details.live_broadcast_content != "none":
        return False
    duration = parse_iso8601_duration(details.duration_iso8601)
    if duration is None:
        return False
    return settings.youtube_min_duration_seconds <= duration <= settings.youtube_max_duration_seconds


def _candidate_dict(details: YoutubeVideoDetails, duration_seconds: int) -> dict[str, Any]:
    """Représentation JSON-sérialisable d'un candidat déjà filtré (ce qui est mis en cache)."""
    return {
        "video_id": details.video_id,
        "title": details.title,
        "channel_title": details.channel_title,
        "duration_seconds": duration_seconds,
        "published_at": details.published_at,
    }


def video_from_candidate(candidate: dict[str, Any]) -> CourseVideo | None:
    """Reconstruit un `CourseVideo` depuis un candidat (cache ou fraîchement filtré)."""
    video_id = candidate.get("video_id", "")
    video = candidate_video(f"https://www.youtube.com/watch?v={video_id}", candidate.get("title", ""))
    if video is None:
        return None
    return video.model_copy(
        update={
            "channel": candidate.get("channel_title") or video.channel,
            "duration_seconds": candidate.get("duration_seconds"),
            "published_at": candidate.get("published_at") or None,
        }
    )


def clean_queries(queries: list[str], max_queries: int) -> list[str]:
    """Nettoie (borne, tronque) et dédoublonne les requêtes, dans l'ordre, jusqu'à `max_queries`."""
    seen: set[str] = set()
    cleaned: list[str] = []
    for raw in queries:
        query = raw.strip()[:_QUERY_MAX_CHARS]
        key = query.casefold()
        if query and key not in seen:
            seen.add(key)
            cleaned.append(query)
        if len(cleaned) >= max(max_queries, 0):
            break
    return cleaned


def fallback_queries(topic: str) -> list[str]:
    """Requêtes déterministes utilisées quand Gemini n'en propose aucune d'exploitable."""
    return [f"{topic} cours", f"{topic} exercices corrigés"]


def round_robin_merge(per_query: list[list[dict[str, Any]]], limit: int) -> list[dict[str, Any]]:
    """Fusionne les candidats de chaque requête à tour de rôle (chacune contribue), dédoublonnés."""
    merged: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for round_items in zip_longest(*per_query):
        for candidate in round_items:
            if candidate is None or candidate["video_id"] in seen_ids:
                continue
            seen_ids.add(candidate["video_id"])
            merged.append(candidate)
            if len(merged) >= limit:
                return merged
    return merged


# ─── YouTube Data API v3 (V1) ─────────────────────────────────


async def find_course_videos(
    queries: list[str],
    topic: str,
    settings: Settings,
    db_session_factory: async_sessionmaker | None = None,
    *,
    client: YoutubeDataClient | None = None,
) -> list[CourseVideo]:
    """Vidéos trouvées via YouTube Data API v3 ; `[]` si pas de clé, quota épuisé, ou aucun résultat
    exploitable — l'appelant tente alors le repli grounding + oEmbed. Ne lève jamais.

    `client` est injectable (tests) ; sinon un `YoutubeDataClient` est créé et fermé ici.
    """
    if not settings.youtube_api_key:
        return []
    if _breaker.is_open():
        logger.info("youtube_quota_breaker_still_open")
        return []

    cleaned = clean_queries(queries, settings.youtube_max_queries) or fallback_queries(topic)
    owns_client = client is None
    data_client = client or YoutubeDataClient(
        settings.youtube_api_key.get_secret_value(),
        timeout_seconds=settings.youtube_timeout_seconds,
        relevance_language=settings.youtube_relevance_language,
        region_code=settings.youtube_region_code,
    )
    ids_by_query: dict[str, list[str]] = {}
    try:
        query_keys = [
            cache_repo.normalize_query_key(
                q, language=settings.youtube_relevance_language, region=settings.youtube_region_code
            )
            for q in cleaned
        ]
        cached: dict[str, list[dict[str, Any]] | None] = {}
        if db_session_factory is not None:
            async with db_session_factory() as db:
                for key in query_keys:
                    cached[key] = await cache_repo.get_fresh(db, key, settings.youtube_cache_ttl)

        missing = [(q, key) for q, key in zip(cleaned, query_keys, strict=True) if cached.get(key) is None]
        fresh_by_query: dict[str, list[dict[str, Any]]] = {}
        quota_units = 0
        if missing:
            for query, _key in missing:
                ids_by_query[query] = await data_client.search(query, max_results=settings.youtube_search_max_results)
                quota_units += 100

            all_ids = [i for ids in ids_by_query.values() for i in ids]
            unique_ids = list(dict.fromkeys(all_ids))
            details_by_id: dict[str, YoutubeVideoDetails] = {}
            if unique_ids:
                details_by_id = {d.video_id: d for d in await data_client.details(unique_ids)}
                quota_units += 1

            if db_session_factory is not None:
                async with db_session_factory() as db:
                    for query, key in missing:
                        filtered = []
                        for vid in ids_by_query.get(query, []):
                            details = details_by_id.get(vid)
                            if details is None or not passes_filters(details, settings):
                                continue
                            filtered.append(_candidate_dict(details, parse_iso8601_duration(details.duration_iso8601) or 0))
                        fresh_by_query[key] = filtered
                        await cache_repo.upsert(db, key, query, filtered)
            else:
                for query, key in missing:
                    fresh_by_query[key] = [
                        _candidate_dict(details, parse_iso8601_duration(details.duration_iso8601) or 0)
                        for vid in ids_by_query.get(query, [])
                        if (details := details_by_id.get(vid)) is not None and passes_filters(details, settings)
                    ]

        per_query = [cached[key] if cached.get(key) is not None else fresh_by_query.get(key, []) for key in query_keys]
        merged = round_robin_merge(per_query, settings.course_videos_max)
        videos = [v for v in (video_from_candidate(c) for c in merged) if v is not None]

        logger.info(
            "course_videos_source",
            extra={
                "source": "cache" if not missing else "data_api",
                "queries": len(cleaned),
                "candidates_before_filters": sum(len(v) for v in ids_by_query.values()),
                "candidates_after_filters": len(merged),
                "quota_units": quota_units,
            },
        )
        return videos
    except YoutubeQuotaExceeded:
        _breaker.trip()
        return []
    except YoutubeServiceError as exc:
        logger.warning("youtube_data_api_failed", extra={"error": str(exc)})
        return []
    finally:
        if owns_client:
            await data_client.close()


# ─── Repli : recherche groundée Gemini + vérification oEmbed (historique) ──


async def _search_videos(topic: str, gemini_client: GeminiClient) -> list[CourseVideo]:
    """Recherche web (grounding) de vidéos YouTube sur le sujet ; URL extraites du texte, non vérifiées."""
    raw, web_sources = await gemini_client.search_grounded(
        prompt=(
            f"Trouve 3 vidéos YouTube pédagogiques, de préférence en français, qui expliquent bien : {topic}. "
            "Donne pour chacune son titre et son URL complète (https://www.youtube.com/watch?v=...). "
            "Ne cite que des vidéos réellement trouvées."
        ),
        system_instruction="Tu es un assistant de recherche de ressources pédagogiques.",
    )
    urls = list(dict.fromkeys(_YOUTUBE_URL_RE.findall(raw)))
    from_sources = await resolve_grounding_video_ids(web_sources)
    urls += [f"https://www.youtube.com/watch?v={vid}" for vid in from_sources]
    return [v for v in (candidate_video(u) for u in dict.fromkeys(urls)) if v]


# ─── Point d'entrée ───────────────────────────────────────────


async def attach_verified_videos(
    response: CourseGenerationResponse,
    settings: Settings,
    gemini_client: GeminiClient | None = None,
    search_queries: list[str] | None = None,
    db_session_factory: async_sessionmaker | None = None,
) -> CourseGenerationResponse:
    """Vidéos du cours : YouTube Data API en priorité, repli grounding + oEmbed. Best-effort, ne lève jamais.

    Paramètres:
        search_queries: `video_search_queries` proposées par Gemini (jamais d'ID/URL — voir schéma).
        db_session_factory: pour le cache de recherche ; sans lui, la Data API est appelée sans cache.
    """
    if not settings.course_videos_enabled:
        return response.model_copy(update={"videos": []})

    topic = f"{response.meta.title} ({response.meta.subject})".strip()
    videos: list[CourseVideo] = []
    try:
        videos = await find_course_videos(search_queries or [], topic, settings, db_session_factory)
        if not videos and gemini_client is not None:
            candidates = await _search_videos(topic, gemini_client)
            videos = await verify_videos(
                candidates,
                timeout_seconds=settings.course_videos_verify_timeout_seconds,
                max_videos=settings.course_videos_max,
            )
            if videos:
                logger.info("course_videos_source", extra={"source": "grounding_fallback"})
    except Exception:
        logger.warning("course_videos_lookup_failed", exc_info=True)
    if not videos:
        logger.warning("course_videos_none_verified")
    return response.model_copy(update={"videos": videos[: settings.course_videos_max]})
