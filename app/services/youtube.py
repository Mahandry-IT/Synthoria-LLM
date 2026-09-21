"""Vérification des vidéos YouTube suggérées par le modèle.

Le modèle peut inventer une URL : chaque candidat est donc validé via l'endpoint
oEmbed de YouTube (sans clé d'API). Une vidéo inexistante, privée ou dont
l'intégration est désactivée renvoie une erreur HTTP et est écartée.
"""

import asyncio
import logging
import re
from urllib.parse import parse_qs, urlparse

import httpx

from app.api.schemas import CourseVideo

logger = logging.getLogger(__name__)

_OEMBED_URL = "https://www.youtube.com/oembed"
_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"}


def extract_video_id(url: str) -> str | None:
    """Extrait l'identifiant d'une URL YouTube (watch, youtu.be, embed, shorts), sinon None."""
    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return None
    host = (parsed.hostname or "").lower()
    candidate: str | None = None
    if host == "youtu.be":
        candidate = parsed.path.lstrip("/").split("/")[0]
    elif host in _YOUTUBE_HOSTS:
        if parsed.path == "/watch":
            candidate = (parse_qs(parsed.query).get("v") or [None])[0]
        else:
            parts = [p for p in parsed.path.split("/") if p]
            if len(parts) >= 2 and parts[0] in {"embed", "shorts", "live"}:
                candidate = parts[1]
    return candidate if candidate and _VIDEO_ID_RE.match(candidate) else None


def candidate_video(url: str, title: str = "") -> CourseVideo | None:
    """Construit une vidéo non vérifiée depuis une URL, ou None si ce n'est pas une vidéo YouTube."""
    video_id = extract_video_id(url)
    if video_id is None:
        return None
    return CourseVideo(
        video_id=video_id,
        url=f"https://www.youtube.com/watch?v={video_id}",
        embed_url=f"https://www.youtube.com/embed/{video_id}",
        thumbnail_url=f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
        title=title.strip() or "Vidéo YouTube",
    )


# Le grounding Gemini cite ses sources via une redirection : seule cette origine est suivie (jamais d'URL libre).
_GROUNDING_REDIRECT_HOST = "vertexaisearch.cloud.google.com"


async def resolve_grounding_video_ids(
    sources: list[dict[str, str]],
    timeout_seconds: float = 5.0,
    max_sources: int = 8,
) -> list[str]:
    """Identifiants de vidéos YouTube trouvés dans les sources de citation d'une recherche groundée.

    Une source peut pointer directement vers YouTube, ou vers une redirection du grounding Gemini
    dont la cible (en-tête `Location`, un seul saut) est la vidéo. Les autres sources sont ignorées.
    Best-effort : aucune exception ne remonte.
    """
    ids: list[str] = []
    pending: list[str] = []
    for source in sources[:max_sources]:
        reference = source.get("reference", "")
        direct = extract_video_id(reference)
        if direct:
            ids.append(direct)
        elif urlparse(reference).hostname == _GROUNDING_REDIRECT_HOST:
            pending.append(reference)

    async def _target(client: httpx.AsyncClient, url: str) -> str | None:
        try:
            response = await client.get(url, follow_redirects=False)
        except httpx.HTTPError as exc:
            logger.info("youtube_grounding_redirect_unreachable", extra={"error": str(exc)})
            return None
        return extract_video_id(response.headers.get("location", ""))

    if pending:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            ids.extend(v for v in await asyncio.gather(*(_target(client, u) for u in pending)) if v)
    return list(dict.fromkeys(ids))


async def _verify_one(client: httpx.AsyncClient, video: CourseVideo) -> CourseVideo | None:
    try:
        response = await client.get(_OEMBED_URL, params={"url": video.url, "format": "json"})
    except httpx.HTTPError as exc:
        logger.warning("youtube_verify_unreachable", extra={"video_id": video.video_id, "error": str(exc)})
        return None
    if response.status_code != 200:
        logger.info("youtube_video_rejected", extra={"video_id": video.video_id, "status": response.status_code})
        return None
    try:
        data = response.json()
    except ValueError:
        return None
    return video.model_copy(update={
        "title": data.get("title") or video.title,
        "channel": data.get("author_name") or video.channel,
    })


async def verify_videos(
    candidates: list[CourseVideo],
    timeout_seconds: float = 5.0,
    max_videos: int = 3,
) -> list[CourseVideo]:
    """Ne conserve que les vidéos réellement disponibles (titre/chaîne remplacés par ceux de YouTube).

    Best-effort : aucune exception ne remonte. Si YouTube est injoignable,
    aucune vidéo n'est retournée plutôt qu'une URL non vérifiée.
    """
    unique = list({v.video_id: v for v in candidates}.values())
    if not unique:
        return []
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        results = await asyncio.gather(*(_verify_one(client, v) for v in unique))
    return [v for v in results if v is not None][:max_videos]
