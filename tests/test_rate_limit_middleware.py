import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.core.rate_limit import WINDOW_SECONDS, RateLimitMiddleware, SlidingWindowLimiter


def _client(limit: int) -> TestClient:
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, requests_per_minute=limit)

    @app.get("/read")
    async def read() -> dict:
        return {"ok": True}

    @app.post("/write")
    async def write() -> dict:
        return {"ok": True}

    return TestClient(app)


def test_write_over_the_limit_returns_429_json_with_retry_after_not_500():
    client = _client(limit=2)
    assert client.post("/write").status_code == 200
    assert client.post("/write").status_code == 200

    response = client.post("/write")

    assert response.status_code == 429
    assert response.json()["detail"] == "Trop de requêtes, réessayez plus tard"
    assert response.json()["error_code"] == "rate_limited"
    retry_after = int(response.headers["Retry-After"])
    assert 1 <= retry_after <= WINDOW_SECONDS


def test_get_requests_are_never_limited():
    client = _client(limit=1)
    for _ in range(5):
        assert client.get("/read").status_code == 200
    # Les lectures ne consomment pas la fenêtre : la première écriture passe encore.
    assert client.post("/write").status_code == 200


def test_sliding_window_limiter_raises_429_with_retry_after():
    limiter = SlidingWindowLimiter()
    limiter.check("k", limit=1)

    with pytest.raises(HTTPException) as exc_info:
        limiter.check("k", limit=1, detail="stop")

    assert exc_info.value.status_code == 429
    assert exc_info.value.detail == "stop"
    assert int(exc_info.value.headers["Retry-After"]) >= 1
