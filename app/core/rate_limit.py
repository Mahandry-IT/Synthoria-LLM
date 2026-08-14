import time
from collections import defaultdict

from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware

WINDOW_SECONDS = 60


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting en mémoire (par IP). Suffisant pour une instance unique.

    ⚠️ Pour un déploiement multi-instance, remplacer par un backend partagé (Redis).
    """

    def __init__(self, app, requests_per_minute: int) -> None:
        super().__init__(app)
        self._limit = requests_per_minute
        self._hits: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()

        self._hits[client_ip] = [t for t in self._hits[client_ip] if now - t < WINDOW_SECONDS]

        if len(self._hits[client_ip]) >= self._limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Trop de requêtes, réessayez plus tard",
            )

        self._hits[client_ip].append(now)
        return await call_next(request)
