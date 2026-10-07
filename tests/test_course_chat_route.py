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

    def active():
        return [m for m in stored if m.deleted_at is None]

    async def get_active(_db, _sid, message_id, for_update=False):
        return next((m for m in active() if m.id == message_id), None)

    async def delete_branch(_db, _sid, message_id, now):
        branch = {message_id}
        for m in stored:  # ordre chronologique : un parent précède ses enfants
            if m.parent_id in branch:
                branch.add(m.id)
        return await mark_deleted([m for m in stored if m.id in branch], now)

    async def mark_deleted(messages, now):
        targets = [m for m in messages if m.deleted_at is None]
        for m in targets:
            m.deleted_at = now
        return len(targets)

    repo = chat_routes.course_chat_repository
    monkeypatch.setattr(repo, "add_exchange", AsyncMock(side_effect=add_exchange))
    monkeypatch.setattr(repo, "count_user_messages_since", AsyncMock(side_effect=count_since))
    monkeypatch.setattr(repo, "list_for_session", AsyncMock(side_effect=lambda *_: active()))
    monkeypatch.setattr(repo, "get_active_message", AsyncMock(side_effect=get_active))
    monkeypatch.setattr(repo, "soft_delete_branch", AsyncMock(side_effect=delete_branch))
    async def delete_all(_db, _sid, now):
        return await mark_deleted(stored, now)

    monkeypatch.setattr(repo, "soft_delete_all", AsyncMock(side_effect=delete_all))
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


# ─── Versions (édition) et suppression ─────────────────────────────────────


def _ask(env, message, **extra):
    res = env.client.post(URL, json={"message": message, **extra})
    assert res.status_code == 200, res.text
    return res.json()


def _sent_history(env) -> list[str]:
    return [turn["text"] for turn in env.gemini.chat.await_args.args[1]]


def test_messages_are_chained_into_a_tree(env):
    first = _ask(env, "q1")
    second = _ask(env, "q2")  # sans parent_id : suite de la dernière réponse

    assert first["user_message"]["parent_id"] is None
    assert first["assistant_message"]["parent_id"] == first["user_message"]["id"]
    assert second["user_message"]["parent_id"] == first["assistant_message"]["id"]
    listed = {m["id"]: m["parent_id"] for m in env.client.get(URL).json()["messages"]}
    assert listed[second["assistant_message"]["id"]] == second["user_message"]["id"]


def test_editing_creates_a_sibling_version_and_sends_only_its_branch(env):
    first = _ask(env, "q1")
    original = _ask(env, "q2 originale")

    edited = _ask(env, "q2 éditée", parent_id=original["user_message"]["parent_id"])

    assert edited["user_message"]["parent_id"] == first["assistant_message"]["id"]
    history = _sent_history(env)
    assert len(history) == 2 and "q1" in history[0]
    assert not any("q2 originale" in text for text in history)
    body = env.client.get(URL).json()
    assert [m["content"] for m in body["messages"] if m["role"] == "user"] == ["q1", "q2 originale", "q2 éditée"]
    assert edited["quota"]["used"] == 3 and body["quota"]["used"] == 3  # une édition consomme le quota


def test_null_parent_id_starts_a_new_root_without_history(env):
    _ask(env, "q1")

    root = _ask(env, "q1 éditée", parent_id=None)

    assert root["user_message"]["parent_id"] is None
    assert _sent_history(env) == []


def test_section_and_parent_are_both_accepted(env):
    first = _ask(env, "q1")

    reply = _ask(env, "q2", parent_id=first["assistant_message"]["id"], section_id="s1")

    assert reply["user_message"]["parent_id"] == first["assistant_message"]["id"]


@pytest.mark.parametrize("target", ["unknown", "user_message", "deleted"])
def test_invalid_parent_is_404_without_calling_gemini(env, target):
    first = _ask(env, "q1")
    env.gemini.chat.reset_mock()
    parent = {
        "unknown": str(uuid.uuid4()),
        "user_message": first["user_message"]["id"],
        "deleted": first["assistant_message"]["id"],
    }[target]
    if target == "deleted":
        assert env.client.delete(f"{URL}/messages/{first['user_message']['id']}").status_code == 204

    res = env.client.post(URL, json={"message": "q2", "parent_id": parent})

    assert res.status_code == 404 and res.json()["detail"] == "Message parent introuvable"
    env.gemini.chat.assert_not_called()
    assert len(env.stored) == 2


def test_parent_deleted_during_the_gemini_call_is_404_and_stores_nothing(env):
    first = _ask(env, "q1")

    async def delete_parent_meanwhile(*_args, **_kwargs):
        for m in env.stored:
            m.deleted_at = NOW
        return ("Réponse", [])

    env.gemini.chat.side_effect = delete_parent_meanwhile

    res = env.client.post(URL, json={"message": "q2", "parent_id": first["assistant_message"]["id"]})

    assert res.status_code == 404
    assert len(env.stored) == 2


def test_delete_message_removes_its_answer_and_descendants_but_keeps_siblings_and_quota(env):
    first = _ask(env, "q1")
    original = _ask(env, "q2")
    _ask(env, "q3")  # suite de q2
    sibling = _ask(env, "q2 éditée", parent_id=first["assistant_message"]["id"])

    res = env.client.delete(f"{URL}/messages/{original['user_message']['id']}")

    assert res.status_code == 204 and res.content == b""
    body = env.client.get(URL).json()
    assert [m["content"] for m in body["messages"] if m["role"] == "user"] == ["q1", "q2 éditée"]
    assert len(body["messages"]) == 4
    assert sibling["assistant_message"]["id"] in {m["id"] for m in body["messages"]}
    assert body["quota"]["used"] == 4  # supprimer ne rend pas de quota


def test_without_parent_id_a_question_follows_the_latest_remaining_answer(env):
    first = _ask(env, "q1")
    second = _ask(env, "q2")
    env.client.delete(f"{URL}/messages/{second['user_message']['id']}")

    third = _ask(env, "q3")

    assert third["user_message"]["parent_id"] == first["assistant_message"]["id"]


@pytest.mark.parametrize("target", ["assistant", "unknown", "already_deleted"])
def test_delete_message_is_404_for_non_question_unknown_or_deleted(env, target):
    first = _ask(env, "q1")
    message_id = {
        "assistant": first["assistant_message"]["id"],
        "unknown": str(uuid.uuid4()),
        "already_deleted": first["user_message"]["id"],
    }[target]
    if target == "already_deleted":
        env.client.delete(f"{URL}/messages/{message_id}")

    res = env.client.delete(f"{URL}/messages/{message_id}")

    assert res.status_code == 404
    if target == "assistant":
        assert len(env.client.get(URL).json()["messages"]) == 2


def test_delete_whole_chat_hides_history_keeps_quota_and_restarts_at_root(env):
    _ask(env, "q1")
    _ask(env, "q2")

    assert env.client.delete(URL).status_code == 204
    body = env.client.get(URL).json()
    assert body["messages"] == [] and body["quota"]["used"] == 2

    restarted = _ask(env, "q3")
    assert restarted["user_message"]["parent_id"] is None
    assert env.client.delete(URL).status_code == 204  # idempotent


def test_quota_counts_deleted_questions(env):
    for i in range(15):
        _ask(env, f"q{i}")
    env.client.delete(URL)

    assert env.client.post(URL, json={"message": "encore"}).status_code == 429


def test_delete_on_unknown_session_is_404(env):
    env.monkeypatch.setattr(chat_routes.course_session_repository, "get_by_id", AsyncMock(return_value=None))

    assert env.client.delete(URL).status_code == 404
    assert env.client.delete(f"{URL}/messages/{uuid.uuid4()}").status_code == 404
