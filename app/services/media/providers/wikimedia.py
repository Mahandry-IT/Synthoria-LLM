"""Client Wikimedia Commons : recherche d'images via l'API MediaWiki (`generator=search`).

Demande systématiquement une miniature rasterisée (`iiurlwidth`) et ne renvoie JAMAIS le fichier
original : ça évite les formats non raster (SVG) et les originaux parfois énormes, tout en gardant
un seul hôte à autoriser (`upload.wikimedia.org`) pour le téléchargement (`fetcher.py`).
"""

from __future__ import annotations

import logging

import httpx

from app.core.exceptions import MediaWebError
from app.services.media.providers import WebImageCandidate

logger = logging.getLogger(__name__)

_API_URL = "https://commons.wikimedia.org/w/api.php"
_NAMESPACE_FILE = 6
_THUMB_WIDTH = 1600  # aligné sur la dimension max de app.services.media.storage.normalize_image


class WikimediaProvider:
    name = "wikimedia"

    def __init__(
        self, *, user_agent: str, timeout_seconds: float = 8.0, client: httpx.AsyncClient | None = None
    ) -> None:
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds, headers={"User-Agent": user_agent})

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def search(self, query: str, *, limit: int) -> list[WebImageCandidate]:
        params = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": query,
            "gsrnamespace": _NAMESPACE_FILE,
            "gsrlimit": limit,
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata",
            "iiurlwidth": _THUMB_WIDTH,
        }
        try:
            response = await self._client.get(_API_URL, params=params)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            raise MediaWebError(f"Wikimedia Commons injoignable : {exc}") from exc

        pages = (data.get("query") or {}).get("pages") or {}
        candidates: list[WebImageCandidate] = []
        for page in pages.values():
            infos = page.get("imageinfo") or []
            if not infos:
                continue
            info = infos[0]
            mime = info.get("mime", "")
            if not mime.startswith("image/"):
                continue
            thumb_url = info.get("thumburl")
            if not thumb_url:
                continue
            meta = info.get("extmetadata") or {}
            title = page.get("title", "")
            candidates.append(
                WebImageCandidate(
                    provider=self.name,
                    thumb_url=thumb_url,
                    download_url=thumb_url,
                    page_url=info.get("descriptionurl") or f"https://commons.wikimedia.org/wiki/{title}",
                    width=int(info.get("thumbwidth") or info.get("width") or 0),
                    height=int(info.get("thumbheight") or info.get("height") or 0),
                    title=title,
                    author=(meta.get("Artist") or {}).get("value"),
                    license=(meta.get("LicenseShortName") or {}).get("value"),
                    license_url=(meta.get("LicenseUrl") or {}).get("value"),
                )
            )
        return candidates
