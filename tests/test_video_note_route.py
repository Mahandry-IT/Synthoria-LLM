import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import routes

VIDEOS = [
    {"video_id": "abc123", "url": "u1", "embed_url": "e1", "thumbnail_url": "t1", "title": "V1", "channel": ""},
    {"video_id": "def456", "url": "u2", "embed_url": "e2", "thumbnail_url": "t2", "title": "V2", "channel": ""},
]


@pytest.fixture
def env(monkeypatch):
    app = FastAPI()
    app.include_router(routes.router)

    @asynccontextmanager
    async def factory():
        yield object()

    app.state.db_session_factory = factory

    row = SimpleNamespace(
        id=uuid.uuid4(), question="Q ?", mode="file_question", filenames=[],
        gemini_response={"videos": [dict(v) for v in VIDEOS]},
    )
    monkeypatch.setattr(routes.course_session_repository, "get_by_id", AsyncMock(return_value=row))
    return SimpleNamespace(client=TestClient(app), monkeypatch=monkeypatch, app=app, row=row)


def _note_url(session_id, video_id: str = "abc123") -> str:
    return f"/courses/{session_id}/videos/{video_id}/note"


def test_save_note_succeeds(env):
    saved = SimpleNamespace(note="À revoir.", updated_at=__import__("datetime").datetime(2026, 1, 1))
    upsert = AsyncMock(return_value=saved)
    env.monkeypatch.setattr(routes.course_video_note_repository, "upsert", upsert)

    res = env.client.put(_note_url(env.row.id, "def456"), json={"note": "À revoir."})

    assert res.status_code == 200
    assert res.json()["note"] == "À revoir."
    upsert.assert_awaited_once()


def test_save_note_unknown_session_is_404(env):
    env.monkeypatch.setattr(routes.course_session_repository, "get_by_id", AsyncMock(return_value=None))

    assert env.client.put(_note_url(uuid.uuid4()), json={"note": "x"}).status_code == 404


def test_save_note_unknown_video_is_404(env):
    assert env.client.put(_note_url(env.row.id, "unknown"), json={"note": "x"}).status_code == 404


def test_save_empty_note_clears_it(env):
    saved = SimpleNamespace(note="", updated_at=__import__("datetime").datetime(2026, 1, 1))
    upsert = AsyncMock(return_value=saved)
    env.monkeypatch.setattr(routes.course_video_note_repository, "upsert", upsert)

    res = env.client.put(_note_url(env.row.id), json={"note": ""})

    assert res.status_code == 200 and res.json()["note"] == ""
    assert upsert.await_args.args[-1] == ""


def test_save_note_too_long_is_422(env):
    assert env.client.put(_note_url(env.row.id), json={"note": "x" * 2001}).status_code == 422


def test_save_note_is_rate_limited(env):
    env.app.dependency_overrides[routes.get_settings] = lambda: SimpleNamespace(course_note_rate_limit_per_minute=1)
    saved = SimpleNamespace(note="x", updated_at=__import__("datetime").datetime(2026, 1, 1))
    env.monkeypatch.setattr(routes.course_video_note_repository, "upsert", AsyncMock(return_value=saved))

    codes = [env.client.put(_note_url(env.row.id), json={"note": "x"}).status_code for _ in range(2)]

    assert codes == [200, 429]
