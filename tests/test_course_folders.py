import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import routes
from app.db.models import DEFAULT_COURSE_FOLDER, DEFAULT_COURSE_SUBFOLDER
from app.repositories import course_session_repository as repo

SID = uuid.uuid4()


# ─── Repository ────────────────────────────────────────────────


def _session(row=None, execute_result=None) -> MagicMock:
    session = MagicMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    if execute_result is not None:
        session.execute = AsyncMock(return_value=execute_result)
    else:
        session.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=lambda: row))
    return session


@pytest.mark.asyncio
async def test_move_to_folder_updates_row_and_commits():
    row = SimpleNamespace(id=SID, folder=DEFAULT_COURSE_FOLDER, subfolder=DEFAULT_COURSE_SUBFOLDER)
    session = _session(row)

    result = await repo.move_to_folder(session, SID, folder="SFI", subfolder="2026")

    assert (result.folder, result.subfolder) == ("SFI", "2026")
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_move_to_folder_returns_none_when_session_missing():
    session = _session(None)

    assert await repo.move_to_folder(session, SID, folder="SFI", subfolder="2026") is None
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_folder_reassigns_to_default_and_returns_moved_count():
    session = _session(execute_result=MagicMock(rowcount=3))

    moved = await repo.delete_folder(session, "SFI")

    assert moved == 3
    session.commit.assert_awaited_once()
    stmt = session.execute.await_args.args[0]
    compiled = stmt.compile(compile_kwargs={"literal_binds": True})
    assert DEFAULT_COURSE_FOLDER in str(compiled)
    assert DEFAULT_COURSE_SUBFOLDER in str(compiled)


@pytest.mark.asyncio
async def test_delete_subfolder_reassigns_to_default_subfolder_only():
    session = _session(execute_result=MagicMock(rowcount=2))

    moved = await repo.delete_subfolder(session, "SFI", "2025")

    assert moved == 2
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_list_folders_returns_grouped_rows():
    rows = [("SFI", "2026", 4), ("SFI", DEFAULT_COURSE_SUBFOLDER, 1), (DEFAULT_COURSE_FOLDER, DEFAULT_COURSE_SUBFOLDER, 2)]
    session = _session(execute_result=MagicMock(all=lambda: rows))

    result = await repo.list_folders(session)

    assert result == rows


# ─── Routes ────────────────────────────────────────────────────


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(routes.router)

    @asynccontextmanager
    async def factory():
        yield object()

    app.state.db_session_factory = factory
    return TestClient(app)


def test_list_folders_aggregates_counts_and_subfolders(client, monkeypatch):
    monkeypatch.setattr(
        routes.course_session_repository,
        "list_folders",
        AsyncMock(return_value=[("SFI", "2026", 4), ("SFI", DEFAULT_COURSE_SUBFOLDER, 1)]),
    )

    res = client.get("/courses/folders")

    assert res.status_code == 200
    body = res.json()
    assert body == [
        {
            "name": "SFI",
            "course_count": 5,
            "subfolders": [
                {"name": "2026", "course_count": 4},
                {"name": DEFAULT_COURSE_SUBFOLDER, "course_count": 1},
            ],
        }
    ]


def test_move_course_folder_success(client, monkeypatch):
    row = SimpleNamespace(
        id=SID, created_at=__import__("datetime").datetime(2026, 1, 1),
        question="Q ?", filenames=[], mode="file_question", folder="SFI", subfolder="2026",
    )
    monkeypatch.setattr(routes.course_session_repository, "move_to_folder", AsyncMock(return_value=row))

    res = client.put(f"/courses/history/{SID}/folder", json={"folder": "SFI", "subfolder": "2026"})

    assert res.status_code == 200
    assert (res.json()["folder"], res.json()["subfolder"]) == ("SFI", "2026")


def test_move_course_folder_blank_subfolder_uses_default(client, monkeypatch):
    mock_move = AsyncMock(return_value=SimpleNamespace(
        id=SID, created_at=__import__("datetime").datetime(2026, 1, 1),
        question="Q ?", filenames=[], mode="file_question", folder="SFI", subfolder=DEFAULT_COURSE_SUBFOLDER,
    ))
    monkeypatch.setattr(routes.course_session_repository, "move_to_folder", mock_move)

    res = client.put(f"/courses/history/{SID}/folder", json={"folder": "SFI"})

    assert res.status_code == 200
    assert mock_move.call_args.kwargs["subfolder"] == DEFAULT_COURSE_SUBFOLDER


def test_move_course_folder_unknown_session_404(client, monkeypatch):
    monkeypatch.setattr(routes.course_session_repository, "move_to_folder", AsyncMock(return_value=None))

    res = client.put(f"/courses/history/{uuid.uuid4()}/folder", json={"folder": "SFI"})

    assert res.status_code == 404


def test_delete_folder_moves_courses_to_default(client, monkeypatch):
    monkeypatch.setattr(routes.course_session_repository, "delete_folder", AsyncMock(return_value=3))

    res = client.delete("/courses/folders/SFI")

    assert res.status_code == 200
    assert res.json() == {"moved": 3}


def test_delete_default_folder_rejected(client):
    res = client.delete(f"/courses/folders/{DEFAULT_COURSE_FOLDER}")

    assert res.status_code == 400


def test_delete_subfolder_moves_courses_to_default_subfolder(client, monkeypatch):
    monkeypatch.setattr(routes.course_session_repository, "delete_subfolder", AsyncMock(return_value=2))

    res = client.delete("/courses/folders/SFI/subfolders/2025")

    assert res.status_code == 200
    assert res.json() == {"moved": 2}


def test_delete_default_subfolder_rejected(client):
    res = client.delete(f"/courses/folders/SFI/subfolders/{DEFAULT_COURSE_SUBFOLDER}")

    assert res.status_code == 400


def test_list_course_history_passes_folder_filters(client, monkeypatch):
    mock_list = AsyncMock(return_value=([], 0))
    monkeypatch.setattr(routes.course_session_repository, "list_paginated", mock_list)

    res = client.get("/courses/history?folder=SFI&subfolder=2026")

    assert res.status_code == 200
    assert mock_list.call_args.kwargs["folder"] == "SFI"
    assert mock_list.call_args.kwargs["subfolder"] == "2026"
