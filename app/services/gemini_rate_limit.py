"""Limite de débit client (RPM) pour l'API Gemini.

Espace les appels au lieu de les envoyer en rafale et de laisser Google répondre 429
(`RESOURCE_EXHAUSTED`) : chaque appelant attend un créneau libre dans la fenêtre glissante
d'une minute au lieu d'échouer puis de retenter.

Best-effort, **par processus** : pas de coordination entre les conteneurs `api` et `worker`
(chacun a sa propre instance de `GeminiClient`, donc sa propre fenêtre). C'est le scénario qui
déclenche la plupart des 429 observés en pratique — plusieurs appels rapprochés dans le même
conteneur (édition du plan, lots de sections d'un même cours) — qui est corrigé ; un usage
concurrent soutenu des deux conteneurs peut encore additionner leurs fenêtres et dépasser le
quota réel de la clé API. Réduire `GEMINI_RPM_LIMIT` en conséquence si c'est observé.
"""

import asyncio
import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

_WINDOW_SECONDS = 60.0


class GeminiRateLimiter:
    """Fenêtre glissante asynchrone : au plus `limit_per_minute` créneaux accordés par minute.

    `clock`/`sleep` sont injectables (tests) ; par défaut `time.monotonic`/`asyncio.sleep`.
    """

    def __init__(
        self,
        limit_per_minute: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._limit = max(1, limit_per_minute)
        self._clock = clock
        self._sleep = sleep
        self._lock = asyncio.Lock()
        self._hits: deque[float] = deque()

    async def acquire(self) -> None:
        """Bloque jusqu'à ce qu'un créneau soit disponible dans la fenêtre, puis le réserve."""
        while True:
            async with self._lock:
                now = self._clock()
                while self._hits and now - self._hits[0] >= _WINDOW_SECONDS:
                    self._hits.popleft()
                if len(self._hits) < self._limit:
                    self._hits.append(now)
                    return
                wait = _WINDOW_SECONDS - (now - self._hits[0]) + 0.05
            logger.info("gemini_rate_limit_throttled", extra={"wait_seconds": round(wait, 2)})
            await self._sleep(max(wait, 0.05))
