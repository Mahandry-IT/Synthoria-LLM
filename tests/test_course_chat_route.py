import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import chat_routes
from app.core.config import Settings
from app.core.exceptions import GeminiInvalidResponseError, GeminiQuotaExceededError, GeminiUnavailableError
from app.db.models import CourseChatMessage

NOW = datetime(2026, 10, 4, 22, 0, 0, tzinfo=timezone.utc)
COURSE = {"meta": {"title": "Le transformateur"}, "summary": "Le rapport fixe la tension."}


def _settings(**overrides) -> Settings:
    values = {"gemini_api_key": "k", "chat_daily_limit": 15, "chat_rate_limit_per_minute": 100, **overrides}
    return Settings(**values)


@pytest.fixture
def env(monkeypatch):
    app = FastAPI()
    app.include_router(chat_routes.router)

    @asynccontextmanager
    async def factory():
        yield object()

    app.state.db_session_factory = factory
    gemini = AsyncMock()
    gemini.chat.return_value = ("Le rapport vaut N2/N1.", [{"type": "web", "label": "A", "reference": "https://a"}])
    app.dependency_overrides[chat_routes.get_gemini_client] = lambda: gemini
    app.dependency_overrides[chat_routes.get_settings] = lambda: _settings()
    monkeypatch.setattr(chat_routes, "_utc_now", lambda: NOW)

    stored: list[CourseChatMessage] = []

    async def add_exchange(_db, user, assistant):
        stored.extend([user, assistant])
        return user, assistant

    async def count_since(_db, _sid, since):
        return sum(1 for m in stored if m.role == "user" and m.created_at >= since)

    repo = chat_routes.course_chat_repository
    monkeypatch.setattr(repo, "add_exchange", AsyncMock(side_effect=add_exchange))
    monkeypatch.setattr(repo, "count_user_messages_since", AsyncMock(side_effect=count_since))
    monkeypatch.setattr(repo, "list_for_session", AsyncMock(side_effect=lambda *_: list(stored)))
    get_by_id = AsyncMock(return_value=SimpleNamespace(gemini_response=COURSE))
    monkeypatch.setattr(chat_routes.course_session_repository, "get_by_id", get_by_id)
    return SimpleNamespace(client=TestClient(app), app=app, gemini=gemini, stored=stored, monkeypatch=monkeypatch)


SID = uuid.uuid4()
URL = f"/courses/{SID}/chat"


def test_post_returns_exchange_and_quota(env):
    res = env.client.post(URL, json={"message": "  Que vaut m ?  "})

    assert res.status_code == 200, res.text
    body = res.json()
    assert body["user_message"]["role"] == "user" and body["user_message"]["content"] == "Que vaut m ?"
    assistant = body["assistant_message"]
    assert assistant["role"] == "assistant" and assistant["status"] == "answered"
    assert assistant["sources"] == [{"label": "A", "reference": "https://a"}]
    uuid.UUID(assistant["id"])
    assert body["quota"] == {"limit": 15, "used": 1, "remaining": 14, "resets_at": "2026-10-05T00:00:00Z"}
    created = [datetime.fromisoformat(m["created_at"]) for m in (body["user_message"], assistant)]
    assert created[0] < created[1]


def test_get_returns_history_in_order_and_quota(env):
    env.client.post(URL, json={"message": "q1"})
    env.client.post(URL, json={"message": "q2"})

    res = env.client.get(URL)

    assert res.status_code == 200
    body = res.json()
    assert [m["content"] for m in body["messages"] if m["role"] == "user"] == ["q1", "q2"]
    assert len(body["messages"]) == 4
    assert body["quota"]["used"] == 2 and body["quota"]["remaining"] == 13


def test_off_topic_reply_is_flagged_on_both_messages(env):
    env.gemini.chat.return_value = ("[[HORS_SUJET]]", [])

    body = env.client.post(URL, json={"message": "Une recette de crêpes ?"}).json()

    assert body["assistant_message"]["content"] == "Cette question ne correspond pas au thème du cours."
    assert body["assistant_message"]["status"] == body["user_message"]["status"] == "off_topic"
    assert body["quota"]["used"] == 1  # un hors-sujet consomme le quota


def test_unknown_session_is_404(env):
    env.monkeypatch.setattr(chat_routes.course_session_repository, "get_by_id", AsyncMock(return_value=None))

    assert env.client.get(URL).status_code == 404
    assert env.client.post(URL, json={"message": "x"}).status_code == 404
    env.gemini.chat.assert_not_called()


@pytest.mark.parametrize("payload", [{"message": ""}, {"message": "   "}, {"message": "x" * 1001}, {}])
def test_rejects_empty_or_too_long_message(env, payload):
    assert env.client.post(URL, json=payload).status_code == 422
    env.gemini.chat.assert_not_called()


def test_daily_quota_returns_429_with_explicit_detail_and_retry_after(env):
    codes = [env.client.post(URL, json={"message": f"q{i}"}).status_code for i in range(15)]
    res = env.client.post(URL, json={"message": "q16"})

    assert codes == [200] * 15
    assert res.status_code == 429
    assert "Limite de 15 messages par jour atteinte pour ce cours" in res.json()["detail"]
    assert res.headers["Retry-After"] == str(2 * 3600)  # 22:00 UTC → minuit UTC
    assert env.gemini.chat.await_count == 15


def test_quota_counts_only_today_utc(env):
    yesterday = datetime(2026, 10, 3, 23, 59, tzinfo=timezone.utc)
    env.stored.extend(
        CourseChatMessage(id=uuid.uuid4(), session_id=SID, role="user", content="old", status="answered",
                          sources=[], created_at=yesterday)
        for _ in range(15)
    )

    assert env.client.post(URL, json={"message": "q"}).status_code == 200


@pytest.mark.parametrize(
    ("error", "code"),
    [(GeminiInvalidResponseError("x"), 502), (GeminiUnavailableError("x"), 503), (GeminiQuotaExceededError("x"), 429)],
)
def test_gemini_failure_is_translated_and_does_not_consume_quota(env, error, code):
    env.gemini.chat.side_effect = error

    assert env.client.post(URL, json={"message": "q"}).status_code == code
    assert env.stored == []
    assert env.client.get(URL).json()["quota"]["used"] == 0


def test_per_minute_limiter_returns_429(env):
    env.app.dependency_overrides[chat_routes.get_settings] = lambda: _settings(chat_rate_limit_per_minute=2)

    codes = [env.client.post(URL, json={"message": "q"}).status_code for _ in range(3)]

    assert codes == [200, 200, 429]


def test_course_is_read_from_database_not_client(env):
    env.client.post(URL, json={"message": "q", "course": {"summary": "piraté"}})

    system = env.gemini.chat.await_args.args[0]
    assert "Le rapport fixe la tension." in system and "piraté" not in system
