"""Client HTTP pour YouTube Data API v3 (recherche + détails de vidéos).

Ne fait AUCUN filtrage ni classement métier — juste l'appel réseau et le mapping des
réponses JSON en objets typés. Les filtres (durée, confidentialité, etc.) et le
classement vivent dans `app/services/course_videos.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from app.core.exceptions import YoutubeConfigError, YoutubeQuotaExceeded, YoutubeUnavailable

_BASE_URL = "https://www.googleapis.com/youtube/v3"
_QUOTA_REASONS = {"quotaExceeded", "dailyLimitExceeded", "rateLimitExceeded"}
_CONFIG_REASONS = {"keyInvalid", "badRequest", "forbidden"}


@dataclass(frozen=True)
class YoutubeVideoDetails:
    """Champs bruts d'un `videos.list` — non filtrés, non interprétés (voir `course_videos.py`)."""

    video_id: str
    title: str
    channel_title: str
    description: str
    published_at: str
    duration_iso8601: str
    embeddable: bool
    privacy_status: str
    live_broadcast_content: str


def _error_reason(response: httpx.Response) -> str | None:
    try:
        errors = response.json().get("error", {}).get("errors", [])
        return errors[0].get("reason") if errors else None
    except (ValueError, IndexError, AttributeError):
        return None


def _raise_for_status(response: httpx.Response, *, action: str) -> None:
    if response.status_code < 400:
        return
    reason = _error_reason(response)
    if response.status_code == 403 and reason in _QUOTA_REASONS:
        raise YoutubeQuotaExceeded(f"Quota YouTube Data API dépassé ({action}): {reason}")
    if response.status_code in (400, 401) or (response.status_code == 403 and reason in _CONFIG_REASONS):
        raise YoutubeConfigError(f"YouTube Data API a rejeté la requête ({action}): {reason or response.status_code}")
    raise YoutubeUnavailable(f"YouTube Data API a répondu {response.status_code} ({action})")


class YoutubeDataClient:
    """Encapsule les deux appels utilisés : `search.list` (IDs) et `videos.list` (détails)."""

    def __init__(
        self,
        api_key: str,
        *,
        timeout_seconds: float = 5.0,
        relevance_language: str = "fr",
        region_code: str = "FR",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._relevance_language = relevance_language
        self._region_code = region_code
        self._client = client or httpx.AsyncClient(base_url=_BASE_URL, timeout=timeout_seconds)

    async def close(self) -> None:
        await self._client.aclose()

    async def _get(self, path: str, params: dict[str, Any], *, action: str) -> dict[str, Any]:
        try:
            response = await self._client.get(
                path, params=params, headers={"X-Goog-Api-Key": self._api_key}
            )
        except httpx.TimeoutException as exc:
            raise YoutubeUnavailable(f"YouTube Data API injoignable ({action}): délai dépassé") from exc
        except httpx.HTTPError as exc:
            raise YoutubeUnavailable(f"YouTube Data API injoignable ({action}): {exc}") from exc
        _raise_for_status(response, action=action)
        return response.json()

    async def search(self, query: str, *, max_results: int = 8) -> list[str]:
        """Identifiants (dans l'ordre de pertinence Google) des vidéos correspondant à `query`."""
        data = await self._get(
            "search",
            {
                "part": "id",
                "type": "video",
                "q": query,
                "maxResults": max_results,
                "relevanceLanguage": self._relevance_language,
                "regionCode": self._region_code,
                "videoEmbeddable": "true",
                "safeSearch": "strict",
            },
            action="search",
        )
        ids = [item.get("id", {}).get("videoId") for item in data.get("items", [])]
        return [i for i in ids if i]

    async def details(self, video_ids: list[str]) -> list[YoutubeVideoDetails]:
        """Détails de ≤ 50 vidéos en **un seul** appel (limite de l'API). IDs en trop sont ignorés."""
        if not video_ids:
            return []
        if len(video_ids) > 50:
            video_ids = video_ids[:50]
        data = await self._get(
            "videos",
            {"part": "snippet,contentDetails,status,statistics", "id": ",".join(video_ids)},
            action="details",
        )
        results = []
        for item in data.get("items", []):
            snippet = item.get("snippet", {})
            status = item.get("status", {})
            content_details = item.get("contentDetails", {})
            video_id = item.get("id")
            if not video_id:
                continue
            results.append(
                YoutubeVideoDetails(
                    video_id=video_id,
                    title=snippet.get("title", ""),
                    channel_title=snippet.get("channelTitle", ""),
                    description=snippet.get("description", ""),
                    published_at=snippet.get("publishedAt", ""),
                    duration_iso8601=content_details.get("duration", ""),
                    embeddable=bool(status.get("embeddable", False)),
                    privacy_status=status.get("privacyStatus", ""),
                    live_broadcast_content=snippet.get("liveBroadcastContent", "none"),
                )
            )
        return results
