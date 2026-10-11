"""Limite de débit client (RPM et TPM) pour l'API Gemini.

Espace les appels au lieu de les envoyer en rafale et de laisser Google répondre 429
(`RESOURCE_EXHAUSTED`) : chaque appelant attend un créneau libre dans la fenêtre glissante
d'une minute au lieu d'échouer puis de retenter.

Deux fenêtres indépendantes : le nombre de requêtes (`acquire`, tous modèles confondus) et les
tokens d'entrée par modèle (`acquire_tokens`) — un seul gros prompt peut épuiser le quota TPM
alors que le RPM est loin d'être atteint.

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
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_WINDOW_SECONDS = 60.0


@dataclass
class TokenReservation:
    """Tokens réservés dans la fenêtre d'un modèle ; `tokens` est corrigé après l'appel (voir
    `GeminiRateLimiter.correct_tokens`)."""

    at: float
    tokens: int


class GeminiRateLimiter:
    """Fenêtre glissante asynchrone : au plus `limit_per_minute` créneaux accordés par minute, et
    au plus `tpm_limits[model]` tokens d'entrée par minute pour les modèles qui y figurent.

    `clock`/`sleep` sont injectables (tests) ; par défaut `time.monotonic`/`asyncio.sleep`.
    """

    def __init__(
        self,
        limit_per_minute: int,
        *,
        tpm_limits: Mapping[str, int] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._limit = max(1, limit_per_minute)
        self._tpm_limits = {model: max(1, limit) for model, limit in (tpm_limits or {}).items()}
        self._clock = clock
        self._sleep = sleep
        self._lock = asyncio.Lock()
        self._hits: deque[float] = deque()
        self._token_hits: dict[str, deque[TokenReservation]] = {}

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

    def has_token_limit(self, model: str) -> bool:
        """True si `model` a un budget TPM : sinon inutile de compter ses tokens."""
        return model in self._tpm_limits

    async def acquire_tokens(self, model: str, tokens: int) -> TokenReservation | None:
        """Bloque jusqu'à ce que `tokens` tiennent dans la fenêtre TPM de `model`, puis les réserve.

        Retour: la réservation (à passer à `correct_tokens` une fois le compte réel connu), ou
            `None` si `model` n'a pas de limite TPM ou si `tokens` est nul.

        Une requête plus grosse que la limite à elle seule ne peut jamais « tenir » : elle passe
        dès que la fenêtre est vide (log `gemini_tpm_oversized`) plutôt que de bloquer sans fin —
        c'est alors Google qui tranche.
        """
        limit = self._tpm_limits.get(model)
        if limit is None or tokens <= 0:
            return None
        while True:
            async with self._lock:
                now = self._clock()
                hits = self._token_hits.setdefault(model, deque())
                while hits and now - hits[0].at >= _WINDOW_SECONDS:
                    hits.popleft()
                if not hits or sum(hit.tokens for hit in hits) + tokens <= limit:
                    if tokens > limit:
                        logger.warning(
                            "gemini_tpm_oversized model=%s tokens=%d limit=%d", model, tokens, limit,
                            extra={"model": model, "tokens": tokens, "limit": limit},
                        )
                    reservation = TokenReservation(at=now, tokens=tokens)
                    hits.append(reservation)
                    return reservation
                wait = _WINDOW_SECONDS - (now - hits[0].at) + 0.05
            logger.info(
                "gemini_tpm_throttled model=%s tokens=%d wait=%.1fs", model, tokens, wait,
                extra={"model": model, "tokens": tokens, "wait_seconds": round(wait, 2)},
            )
            await self._sleep(max(wait, 0.05))

    def correct_tokens(self, reservation: TokenReservation | None, actual_tokens: int) -> None:
        """Remplace l'estimation réservée par le compte réel renvoyé par Gemini (`usage_metadata`)."""
        if reservation is not None and actual_tokens >= 0:
            reservation.tokens = actual_tokens
