"""Cache best-effort des réponses Gemini (`reformulate_query`/`describe_images`/`rank_images`),
partagé entre les process `api` et `worker` via Postgres (`gemini_response_cache_repository.py`).

Toute opération DB est best-effort : une erreur (Postgres indisponible, etc.) est logguée et
n'empêche jamais un appel Gemini — ce cache est une optimisation de coût, jamais une dépendance
de la génération de cours.
"""

import logging

from sqlalchemy.ext.asyncio import async_sessionmaker

from app.core.config import Settings
from app.repositories import gemini_response_cache_repository as repo

logger = logging.getLogger(__name__)


class GeminiResponseCacheManager:
    """`session_factory=None` : manager no-op (toujours un cache miss) — utile pour instancier
    `GeminiClient` dans des contextes sans DB (scripts ponctuels, tests)."""

    def __init__(self, settings: Settings, session_factory: async_sessionmaker | None) -> None:
        self._settings = settings
        self._session_factory = session_factory

    async def get(self, query_hash: str) -> dict | None:
        if self._session_factory is None:
            return None
        try:
            async with self._session_factory() as session:
                return await repo.get_fresh(session, query_hash, self._settings.gemini_response_cache_ttl)
        except Exception as exc:  # noqa: BLE001 - ne doit jamais bloquer un appel Gemini
            logger.warning("gemini_response_cache_read_failed", extra={"error": str(exc)})
            return None

    async def set(self, query_hash: str, method: str, response: dict) -> None:
        if self._session_factory is None:
            return
        try:
            async with self._session_factory() as session:
                await repo.upsert(session, query_hash, method, response)
        except Exception as exc:  # noqa: BLE001
            logger.warning("gemini_response_cache_write_failed", extra={"error": str(exc)})
