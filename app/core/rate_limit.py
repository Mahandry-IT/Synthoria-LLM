import math
import time
from collections import defaultdict

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

WINDOW_SECONDS = 60
RATE_LIMITED_DETAIL = "Trop de requêtes, réessayez plus tard"
# Lectures peu coûteuses (et polling du Studio) : derrière le proxy Next toutes les requêtes
# partagent l'IP du conteneur, les compter épuiserait la limite globale au moindre écran ouvert.
# Les écritures coûteuses ont en plus leurs propres `SlidingWindowLimiter` et le quota Gemini.
_UNLIMITED_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def _retry_after_seconds(oldest_hit: float, now: float) -> int:
    """Secondes avant que le plus ancien appel de la fenêtre n'en sorte (au moins 1)."""
    return max(1, math.ceil(WINDOW_SECONDS - (now - oldest_hit)))


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting en mémoire (par IP) des requêtes d'écriture. Suffisant pour une instance unique.

    Renvoie directement une `JSONResponse` 429 : une `HTTPException` levée depuis un
    `BaseHTTPMiddleware` n'est pas convertie par Starlette et finit en 500.

    ⚠️ Pour un déploiement multi-instance, remplacer par un backend partagé (Redis).
    """

    def __init__(self, app, requests_per_minute: int) -> None:
        super().__init__(app)
        self._limit = requests_per_minute
        self._hits: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        if request.method in _UNLIMITED_METHODS:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        recent = [t for t in self._hits[client_ip] if now - t < WINDOW_SECONDS]

        if len(recent) >= self._limit:
            self._hits[client_ip] = recent
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": RATE_LIMITED_DETAIL},
                headers={"Retry-After": str(_retry_after_seconds(recent[0], now))},
            )

        recent.append(now)
        self._hits[client_ip] = recent
        return await call_next(request)


class SlidingWindowLimiter:
    """Fenêtre glissante par clé (mémoire) pour limiter un endpoint coûteux, en plus du middleware global."""

    def __init__(self) -> None:
        self._hits: dict[str, list[float]] = defaultdict(list)

    def check(self, key: str, limit: int, detail: str = RATE_LIMITED_DETAIL) -> None:
        now = time.monotonic()
        recent = [t for t in self._hits[key] if now - t < WINDOW_SECONDS]
        if len(recent) >= limit:
            self._hits[key] = recent
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=detail,
                headers={"Retry-After": str(_retry_after_seconds(recent[0], now))},
            )
        recent.append(now)
        self._hits[key] = recent
