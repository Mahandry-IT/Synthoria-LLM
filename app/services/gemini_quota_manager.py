"""Disjoncteur de quota Gemini par modèle, avec vue partagée Postgres entre les process `api` et
`worker` (`app/repositories/gemini_model_quota_repository.py`) et un cache mémoire courte durée
(`gemini_quota_cache_ttl_seconds`) pour éviter une requête DB à chaque appel Gemini.

Toute opération DB est best-effort : une erreur (Postgres indisponible, etc.) est logguée et
n'empêche jamais un appel Gemini — la génération de cours ne doit jamais dépendre de cette
optimisation. La fenêtre de cache (quelques dizaines de secondes) est une fuite de budget bornée
assumée : ce n'est pas une coordination parfaite entre process, seulement une forte réduction du
gaspillage constaté (voir README, section « Limite de débit Gemini »).
"""

import logging
import time
from datetime import date, datetime, timezone

from sqlalchemy.ext.asyncio import async_sessionmaker

from app.core.config import Settings
from app.repositories import gemini_model_quota_repository as repo
from app.services.quota_reset import next_pacific_midnight_utc

logger = logging.getLogger(__name__)


class GeminiQuotaManager:
    """Vue par-modèle de la disponibilité du quota Gemini (`session_factory=None` : manager
    no-op, toujours disponible — utile pour instancier `GeminiClient` dans des contextes sans DB,
    ex. scripts ponctuels ou tests)."""

    def __init__(self, settings: Settings, session_factory: async_sessionmaker | None) -> None:
        self._settings = settings
        self._session_factory = session_factory
        # model -> (horodatage monotonic de mise en cache, disponible)
        self._cache: dict[str, tuple[float, bool]] = {}

    async def is_available(self, model: str) -> bool:
        """False si `model` est marqué épuisé (429 jour reçu) ou a atteint son budget RPD
        configuré aujourd'hui. True par défaut, y compris si la DB est indisponible."""
        if self._session_factory is None:
            return True

        cached = self._cache.get(model)
        if cached is not None and time.monotonic() - cached[0] < self._settings.gemini_quota_cache_ttl_seconds:
            return cached[1]

        available = True
        try:
            async with self._session_factory() as session:
                states = await repo.get_states(session, [model], date.today())
            state = states.get(model)
            if state is not None:
                if state.exhausted_until is not None and state.exhausted_until > datetime.now(timezone.utc):
                    available = False
                else:
                    limit = self._settings.gemini_model_rpd_limits.get(model)
                    if limit is not None and state.day == date.today() and state.request_count >= limit:
                        available = False
        except Exception as exc:  # noqa: BLE001 - ne doit jamais bloquer un appel Gemini
            logger.warning("gemini_quota_manager_db_unavailable", extra={"model": model, "error": str(exc)})

        self._cache[model] = (time.monotonic(), available)
        return available

    async def record_success(self, model: str) -> None:
        """Incrémente le compteur du jour pour `model`, best-effort."""
        if self._session_factory is None:
            return
        try:
            async with self._session_factory() as session:
                await repo.increment(session, model, date.today())
        except Exception as exc:  # noqa: BLE001
            logger.warning("gemini_quota_manager_record_success_failed", extra={"model": model, "error": str(exc)})

    async def record_daily_exhausted(self, model: str) -> None:
        """Marque `model` épuisé jusqu'au prochain minuit Pacifique, best-effort. Met aussi à
        jour le cache local immédiatement (pas d'attente du prochain refresh TTL pour CE
        process : l'autre process, lui, attendra la fin de son propre TTL)."""
        self._cache[model] = (time.monotonic(), False)
        if self._session_factory is None:
            return
        try:
            async with self._session_factory() as session:
                await repo.mark_exhausted(session, model, next_pacific_midnight_utc())
        except Exception as exc:  # noqa: BLE001
            logger.warning("gemini_quota_manager_mark_exhausted_failed", extra={"model": model, "error": str(exc)})
