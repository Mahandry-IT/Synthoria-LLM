import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import routes
from app.core.exceptions import GeminiInvalidResponseError, GeminiQuotaExceededError, GeminiUnavailableError

SECTION = {
    "id": "0", "title": "Le transformateur",
    "challenge": "Que se passe-t-il si on double les spires ?",
    "challenge_key_points": ["la tension double"],
}
NO_CHALLENGE = {"id": "1", "title": "Ancienne section sans défi", "challenge": ""}


@pytest.fixture
def env(monkeypatch):
    app = FastAPI()
    app.include_router(routes.router)

    @asynccontextmanager
    async def factory():
        yield object()

    app.state.db_session_factory = factory
    gemini = AsyncMock()
    gemini.format_structured.return_value = {"verdict": "on_track", "feedback": "Bien vu.", "hint": "Et le courant ?"}
    app.dependency_overrides[routes.get_gemini_client] = lambda: gemini
    row = SimpleNamespace(gemini_response={"sections": [SECTION, NO_CHALLENGE]})
    get_by_id = AsyncMock(return_value=row)
    monkeypatch.setattr(routes.course_session_repository, "get_by_id", get_by_id)
    return SimpleNamespace(client=TestClient(app), gemini=gemini, monkeypatch=monkeypatch, app=app)


def _url(section_id: str = "0") -> str:
    return f"/courses/{uuid.uuid4()}/sections/{section_id}/challenge"


def test_challenge_returns_verdict_feedback_and_hint(env):
    res = env.client.post(_url(), json={"answer": "La tension double."})

    assert res.status_code == 200
    assert res.json() == {"verdict": "on_track", "feedback": "Bien vu.", "hint": "Et le courant ?"}


def test_challenge_uses_section_from_database_not_client(env):
    env.client.post(_url(), json={"answer": "x", "challenge_key_points": ["piraté"], "challenge": "autre"})

    prompt = env.gemini.format_structured.await_args.kwargs["raw_answer"]
    assert "la tension double" in prompt and "piraté" not in prompt and "autre" not in prompt


def test_challenge_unknown_session_section_or_missing_challenge_is_404(env):
    assert env.client.post(_url("42"), json={"answer": "x"}).status_code == 404
    assert env.client.post(_url("1"), json={"answer": "x"}).status_code == 404  # pas de défi
    env.monkeypatch.setattr(routes.course_session_repository, "get_by_id", AsyncMock(return_value=None))
    assert env.client.post(_url(), json={"answer": "x"}).status_code == 404
    env.gemini.format_structured.assert_not_called()


@pytest.mark.parametrize("payload", [{"answer": ""}, {"answer": "   "}, {"answer": "x" * 1001}, {}])
def test_challenge_rejects_empty_or_too_long_answer(env, payload):
    assert env.client.post(_url(), json=payload).status_code == 422
    env.gemini.format_structured.assert_not_called()


def test_challenge_is_rate_limited(env):
    env.app.dependency_overrides[routes.get_settings] = lambda: SimpleNamespace(challenge_rate_limit_per_minute=2)
    codes = [env.client.post(_url(), json={"answer": "x"}).status_code for _ in range(3)]

    assert codes == [200, 200, 429]


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (GeminiInvalidResponseError("x"), 502),
        (GeminiUnavailableError("x"), 503),
        (GeminiQuotaExceededError("x"), 429),
    ],
)
def test_challenge_gemini_errors_are_translated(env, error, code):
    env.gemini.format_structured.side_effect = error
    assert env.client.post(_url(), json={"answer": "x"}).status_code == code


def test_challenge_answer_is_not_persisted(env):
    """Analyse à la volée : aucune écriture en base (seule la lecture de la session est faite)."""
    for name in ("update_section", "append_sections", "save"):
        env.monkeypatch.setattr(routes.course_session_repository, name, AsyncMock())

    env.client.post(_url(), json={"answer": "ma réponse"})

    for name in ("update_section", "append_sections", "save"):
        getattr(routes.course_session_repository, name).assert_not_called()
