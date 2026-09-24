"""Client Openverse : recherche d'images via l'API v1, authentification OAuth2 optionnelle.

Sans `OPENVERSE_CLIENT_ID`/`OPENVERSE_CLIENT_SECRET`, fonctionne en quota anonyme (repli
automatique aussi si l'authentification échoue). Renvoie toujours l'endpoint `thumbnail`
(hébergé par `api.openverse.org`), jamais `url` (hôte d'origine variable — Flickr, musées...) :
garde un seul hôte à autoriser pour le téléchargement (`fetcher.py`).
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from app.core.exceptions import MediaWebError
from app.services.media.providers import WebImageCandidate

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.openverse.org/v1"
_ALLOWED_LICENSES = "cc0,pdm,by,by-sa"
_TOKEN_SAFETY_MARGIN_SECONDS = 30


class OpenverseProvider:
    name = "openverse"

    def __init__(
        self,
        *,
        user_agent: str,
        client_id: str | None = None,
        client_secret: str | None = None,
        timeout_seconds: float = 8.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=_BASE_URL, timeout=timeout_seconds, headers={"User-Agent": user_agent}
        )
        self._client_id = client_id
        self._client_secret = client_secret
        self._token: str | None = None
        self._token_expires_at: float = 0.0

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _auth_headers(self) -> dict[str, str]:
        if not self._client_id or not self._client_secret:
            return {}
        if self._token and time.monotonic() < self._token_expires_at:
            return {"Authorization": f"Bearer {self._token}"}
        try:
            response = await self._client.post(
                "/auth_tokens/token/",
                data={
                    "grant_type": "client_credentials",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                },
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            logger.warning("openverse_auth_failed", extra={"error": str(exc)})
            return {}  # repli sur le quota anonyme, jamais fatal
        token = data.get("access_token")
        if not token:
            return {}
        self._token = token
        self._token_expires_at = time.monotonic() + max(0, int(data.get("expires_in", 0)) - _TOKEN_SAFETY_MARGIN_SECONDS)
        return {"Authorization": f"Bearer {token}"}

    async def search(self, query: str, *, limit: int) -> list[WebImageCandidate]:
        headers = await self._auth_headers()
        params: dict[str, Any] = {
            "q": query, "license": _ALLOWED_LICENSES, "page_size": limit, "mature": "false",
        }
        try:
            response = await self._client.get("/images/", params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            raise MediaWebError(f"Openverse injoignable : {exc}") from exc

        candidates: list[WebImageCandidate] = []
        for item in data.get("results") or []:
            thumb_url = item.get("thumbnail")
            if not thumb_url:
                continue
            candidates.append(
                WebImageCandidate(
                    provider=self.name,
                    thumb_url=thumb_url,
                    download_url=thumb_url,
                    page_url=item.get("foreign_landing_url", ""),
                    width=int(item.get("width") or 0),
                    height=int(item.get("height") or 0),
                    title=item.get("title", ""),
                    author=item.get("creator"),
                    license=item.get("license"),
                    license_url=item.get("license_url"),
                )
            )
        return candidates
