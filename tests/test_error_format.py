"""Format d'erreur unique `{detail, error_code, request_id}` et `debug` réservé au développement."""

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from app.api.routes import _gemini_http_errors
from app.core.errors import ApiError, ErrorCode, register_error_handlers, validation_detail
from app.core.exceptions import GeminiQuotaExceededError
from app.core.rate_limit import RateLimitMiddleware
from app.core.request_id import RequestIdMiddleware


class _Body(BaseModel):
    answer: str = Field(min_length=1)


def _app(*, include_debug: bool, rate_limit: int = 1000) -> TestClient:
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, requests_per_minute=rate_limit)
    app.add_middleware(RequestIdMiddleware)
    register_error_handlers(app, include_debug=include_debug)

    @app.get("/missing")
    async def missing() -> None:
        raise HTTPException(status_code=404, detail="Cours introuvable")

    @app.get("/coded")
    async def coded() -> None:
        raise ApiError(409, "Tentative déjà soumise", ErrorCode.CONFLICT, debug={"type": "X", "message": "m"})

    @app.post("/validate")
    async def validate(body: _Body) -> dict:
        return {"ok": True}

    @app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("secret interne")

    @app.get("/gemini")
    async def gemini() -> None:
        with _gemini_http_errors():
            raise GeminiQuotaExceededError("Quota Gemini dépassé: RESOURCE_EXHAUSTED brut de Google")

    return TestClient(app, raise_server_exceptions=False)


def test_http_exception_has_detail_error_code_and_request_id():
    res = _app(include_debug=False).get("/missing")

    body = res.json()
    assert res.status_code == 404
    assert body["detail"] == "Cours introuvable"
    assert body["error_code"] == "not_found"
    assert body["request_id"] == res.headers["X-Request-ID"]
    assert "debug" not in body


def test_unknown_route_is_also_formatted():
    body = _app(include_debug=False).get("/nope").json()
    assert body["error_code"] == "not_found" and body["detail"] == "Ressource introuvable"


def test_incoming_request_id_is_reused_and_invalid_one_is_replaced():
    client = _app(include_debug=False)

    assert client.get("/missing", headers={"X-Request-ID": "abc-123"}).json()["request_id"] == "abc-123"
    replaced = client.get("/missing", headers={"X-Request-ID": "bad id\twith spaces"})
    assert replaced.json()["request_id"] != "bad id\twith spaces"
    assert len(replaced.json()["request_id"]) == 32


def test_explicit_error_code_is_kept_and_debug_only_in_development():
    prod = _app(include_debug=False).get("/coded").json()
    dev = _app(include_debug=True).get("/coded").json()

    assert prod["error_code"] == dev["error_code"] == "conflict"
    assert "debug" not in prod
    assert dev["debug"] == {"type": "X", "message": "m"}


def test_validation_error_is_readable_with_field_details_in_debug():
    prod = _app(include_debug=False).post("/validate", json={"answer": ""})
    dev = _app(include_debug=True).post("/validate", json={"answer": ""}).json()

    assert prod.status_code == 422
    assert prod.json()["error_code"] == "validation_error"
    assert prod.json()["detail"] == "Requête invalide : vérifiez le champ « answer »"
    assert "debug" not in prod.json()
    assert dev["debug"]["errors"][0]["loc"] == ["body", "answer"]


def test_unhandled_exception_never_leaks_its_message_in_production():
    prod = _app(include_debug=False).get("/boom")
    dev = _app(include_debug=True).get("/boom").json()

    assert prod.status_code == 500
    assert prod.json()["error_code"] == "internal_error"
    assert "secret interne" not in prod.text
    assert dev["debug"] == {"type": "RuntimeError", "message": "secret interne"}


def test_gemini_quota_is_mapped_to_a_user_message_and_stable_code():
    res = _app(include_debug=False).get("/gemini")

    assert res.status_code == 429
    assert res.json()["error_code"] == "gemini_quota"
    assert res.json()["detail"] == "Quota Gemini atteint."
    assert "RESOURCE_EXHAUSTED" not in res.text
    assert res.headers["Retry-After"]


def test_global_rate_limiter_uses_the_common_format():
    client = _app(include_debug=False, rate_limit=1)
    client.post("/validate", json={"answer": "x"})

    res = client.post("/validate", json={"answer": "x"})

    assert res.status_code == 429
    assert res.json()["error_code"] == "rate_limited"
    assert res.json()["request_id"] == res.headers["X-Request-ID"]
    assert res.headers["Retry-After"]


@pytest.mark.parametrize(
    "errors, expected",
    [
        ([], "Requête invalide"),
        ([{"loc": ("body",)}], "Requête invalide : vérifiez le champ « requête »"),
        (
            [{"loc": ("body", "answers", 0)}, {"loc": ("query", "limit")}],
            "Requête invalide : vérifiez les champs « answers.0 », « limit »",
        ),
    ],
)
def test_validation_detail(errors, expected):
    assert validation_detail(errors) == expected
