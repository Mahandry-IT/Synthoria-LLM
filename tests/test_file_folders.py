import io
from contextlib import asynccontextmanager
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.dialects import postgresql

from app.api import routes
from app.db.models import DEFAULT_COURSE_FOLDER, DEFAULT_COURSE_SUBFOLDER
from app.repositories import ingested_file_repository as repo


def _session(*results) -> MagicMock:
    session = MagicMock()
    session.commit = AsyncMock()
    session.execute = AsyncMock(side_effect=list(results))
    return session


def _sql(stmt) -> str:
    return str(stmt.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))


# ─── Repository ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_placements_maps_rows_in_a_single_query():
    session = _session(MagicMock(all=lambda: [("a.pdf", "SFI", "2026")]))

    result = await repo.get_placements(session, ["a.pdf", "b.pdf"])

    assert result == {"a.pdf": ("SFI", "2026")}
    assert session.execute.await_count == 1


@pytest.mark.asyncio
async def test_get_placements_without_filenames_skips_query():
    session = _session()

    assert await repo.get_placements(session, []) == {}
    session.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_move_upserts_with_new_folder_and_default_subfolder():
    session = _session(
        MagicMock(first=lambda: None),  # resolve_casing : dossier inédit
        MagicMock(),  # upsert
    )

    result = await repo.move(session, "a.pdf", folder="SFI", subfolder=None)

    assert result == ("SFI", DEFAULT_COURSE_SUBFOLDER)
    upsert = _sql(session.execute.await_args_list[-1].args[0])
    assert "ON CONFLICT (filename) DO UPDATE" in upsert
    assert "'a.pdf'" in upsert and "'SFI'" in upsert
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_move_reuses_existing_casing():
    session = _session(
        MagicMock(first=lambda: SimpleNamespace(folder="Sfi", n=3, first=datetime(2026, 1, 1))),
        MagicMock(first=lambda: SimpleNamespace(subfolder="Cours", n=2, first=datetime(2026, 1, 1))),
        MagicMock(),
    )

    result = await repo.move(session, "a.pdf", folder="sfi", subfolder="COURS")

    assert result == ("Sfi", "Cours")
    casing_query = _sql(session.execute.await_args_list[0].args[0])
    assert "lower(ingested_files.folder) = 'sfi'" in casing_query


@pytest.mark.asyncio
async def test_move_many_deduplicates_filenames():
    session = _session(MagicMock(first=lambda: None), MagicMock())

    await repo.move_many(session, ["a.pdf", "a.pdf", "b.pdf"], folder="SFI", subfolder=None)

    upsert = _sql(session.execute.await_args_list[-1].args[0])
    assert upsert.count("'a.pdf'") == 1


@pytest.mark.asyncio
async def test_delete_removes_row_and_commits():
    session = _session(MagicMock())

    await repo.delete(session, "a.pdf")

    assert "DELETE FROM ingested_files" in _sql(session.execute.await_args.args[0])
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_folder_reassigns_to_default_case_insensitively():
    session = _session(MagicMock(rowcount=3))

    moved = await repo.delete_folder(session, "SFI")

    assert moved == 3
    sql = _sql(session.execute.await_args.args[0])
    assert "lower(ingested_files.folder) = 'sfi'" in sql
    assert DEFAULT_COURSE_FOLDER in sql and DEFAULT_COURSE_SUBFOLDER in sql
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_subfolder_reassigns_to_default_subfolder_only():
    session = _session(MagicMock(rowcount=2))

    moved = await repo.delete_subfolder(session, "SFI", "2025")

    assert moved == 2
    sql = _sql(session.execute.await_args.args[0])
    assert "lower(ingested_files.subfolder) = '2025'" in sql
    assert DEFAULT_COURSE_FOLDER not in sql


# ─── Routes ────────────────────────────────────────────────────


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(routes.router)

    @asynccontextmanager
    async def factory():
        yield object()

    app.state.db_session_factory = factory
    app.state.vector_store = MagicMock()
    app.state.gemini_client = AsyncMock()
    return TestClient(app)


def test_list_files_merges_placements_and_defaults_files_without_row(client, monkeypatch):
    client.app.state.vector_store.list_files.return_value = [
        {"id": 1, "filename": "a.pdf"},
        {"id": 2, "filename": "b.pdf"},
    ]
    mock_get = AsyncMock(return_value={"a.pdf": ("SFI", "2026")})
    monkeypatch.setattr(routes.ingested_file_repository, "get_placements", mock_get)

    res = client.get("/pdf/files")

    assert res.status_code == 200
    assert res.json()["data"] == [
        {"id": 1, "filename": "a.pdf", "folder": "SFI", "subfolder": "2026"},
        {"id": 2, "filename": "b.pdf", "folder": DEFAULT_COURSE_FOLDER, "subfolder": DEFAULT_COURSE_SUBFOLDER},
    ]
    mock_get.assert_awaited_once()
    assert list(mock_get.await_args.args[1]) == ["a.pdf", "b.pdf"]


def test_list_files_empty_store_skips_database(client, monkeypatch):
    client.app.state.vector_store.list_files.return_value = []
    mock_get = AsyncMock()
    monkeypatch.setattr(routes.ingested_file_repository, "get_placements", mock_get)

    res = client.get("/pdf/files")

    assert res.status_code == 200
    assert res.json()["data"] == []
    mock_get.assert_not_awaited()


def test_move_file_folder_success(client, monkeypatch):
    client.app.state.vector_store.has_file.return_value = True
    mock_move = AsyncMock(return_value=("Sfi", "2026"))
    monkeypatch.setattr(routes.ingested_file_repository, "move", mock_move)

    res = client.put("/pdf/files/cours%20n%C2%B01.pdf/folder", json={"folder": "  sfi ", "subfolder": "2026"})

    assert res.status_code == 200
    assert res.json() == {"filename": "cours n°1.pdf", "folder": "Sfi", "subfolder": "2026"}
    assert mock_move.await_args.args[1] == "cours n°1.pdf"
    assert mock_move.await_args.kwargs == {"folder": "sfi", "subfolder": "2026"}


def test_move_file_folder_blank_subfolder_passes_none(client, monkeypatch):
    client.app.state.vector_store.has_file.return_value = True
    mock_move = AsyncMock(return_value=("SFI", DEFAULT_COURSE_SUBFOLDER))
    monkeypatch.setattr(routes.ingested_file_repository, "move", mock_move)

    res = client.put("/pdf/files/a.pdf/folder", json={"folder": "SFI", "subfolder": "   "})

    assert res.status_code == 200
    assert mock_move.await_args.kwargs["subfolder"] is None
    assert res.json()["subfolder"] == DEFAULT_COURSE_SUBFOLDER


def test_move_unknown_file_404_without_db_write(client, monkeypatch):
    client.app.state.vector_store.has_file.return_value = False
    mock_move = AsyncMock()
    monkeypatch.setattr(routes.ingested_file_repository, "move", mock_move)

    res = client.put("/pdf/files/ghost.pdf/folder", json={"folder": "SFI"})

    assert res.status_code == 404
    mock_move.assert_not_awaited()


@pytest.mark.parametrize("body", [{"folder": "   "}, {"folder": ""}, {"folder": "x" * 201}, {}])
def test_move_file_folder_invalid_body_422(client, body):
    client.app.state.vector_store.has_file.return_value = True

    res = client.put("/pdf/files/a.pdf/folder", json=body)

    assert res.status_code == 422


def test_delete_file_folder_moves_files_to_default(client, monkeypatch):
    monkeypatch.setattr(routes.ingested_file_repository, "delete_folder", AsyncMock(return_value=4))

    res = client.delete("/pdf/folders/SFI")

    assert res.status_code == 200
    assert res.json() == {"moved": 4}


def test_delete_default_file_folder_rejected_case_insensitively(client, monkeypatch):
    mock_delete = AsyncMock()
    monkeypatch.setattr(routes.ingested_file_repository, "delete_folder", mock_delete)

    res = client.delete(f"/pdf/folders/{DEFAULT_COURSE_FOLDER.upper()}")

    assert res.status_code == 400
    mock_delete.assert_not_awaited()


def test_delete_file_subfolder_moves_files_to_default_subfolder(client, monkeypatch):
    monkeypatch.setattr(routes.ingested_file_repository, "delete_subfolder", AsyncMock(return_value=2))

    res = client.delete("/pdf/folders/SFI/subfolders/2025")

    assert res.status_code == 200
    assert res.json() == {"moved": 2}


def test_delete_default_file_subfolder_rejected(client):
    res = client.delete(f"/pdf/folders/SFI/subfolders/{DEFAULT_COURSE_SUBFOLDER}")

    assert res.status_code == 400


def test_delete_file_also_removes_its_folder_row(client, monkeypatch):
    client.app.state.vector_store.remove_file.return_value = 3
    mock_delete = AsyncMock()
    monkeypatch.setattr(routes.ingested_file_repository, "delete", mock_delete)

    res = client.delete("/pdf/files/a.pdf")

    assert res.status_code == 204
    assert mock_delete.await_args.args[1] == "a.pdf"


# ─── Ingestion directe dans un dossier ─────────────────────────


def _pdf(name: str):
    return ("files", (name, io.BytesIO(b"%PDF-1.4 fake"), "application/pdf"))


def _chunks(name: str):
    return [{"id": f"{name}::0::0", "content": "c", "metadata": {"filename": name, "page": 1}}]


def test_ingest_places_ingested_files_in_target_folder(client, monkeypatch):
    store = client.app.state.vector_store
    store.has_file.side_effect = lambda f: f == "dup.pdf"
    store.add_chunks = AsyncMock(return_value=1)
    mock_move_many = AsyncMock()
    monkeypatch.setattr(routes.ingested_file_repository, "move_many", mock_move_many)

    with patch("app.api.routes.extract_pdf_chunks", AsyncMock(side_effect=lambda c, n, *a: _chunks(n))):
        res = client.post(
            "/pdf/ingest",
            files=[_pdf("new.pdf"), _pdf("dup.pdf")],
            data={"folder": " SFI ", "subfolder": "2026"},
        )

    assert res.status_code == 200
    assert mock_move_many.await_args.args[1] == ["new.pdf"]  # le doublon n'est pas déplacé
    assert mock_move_many.await_args.kwargs == {"folder": "SFI", "subfolder": "2026"}


def test_ingest_without_folder_creates_no_row(client, monkeypatch):
    store = client.app.state.vector_store
    store.has_file.return_value = False
    store.add_chunks = AsyncMock(return_value=1)
    mock_move_many = AsyncMock()
    monkeypatch.setattr(routes.ingested_file_repository, "move_many", mock_move_many)

    with patch("app.api.routes.extract_pdf_chunks", AsyncMock(return_value=_chunks("a.pdf"))):
        res = client.post("/pdf/ingest", files=[_pdf("a.pdf")])

    assert res.status_code == 200
    assert res.json()["status"] == "ok"
    mock_move_many.assert_not_awaited()


def test_ingest_subfolder_only_targets_default_folder(client, monkeypatch):
    store = client.app.state.vector_store
    store.has_file.return_value = False
    store.add_chunks = AsyncMock(return_value=1)
    mock_move_many = AsyncMock()
    monkeypatch.setattr(routes.ingested_file_repository, "move_many", mock_move_many)

    with patch("app.api.routes.extract_pdf_chunks", AsyncMock(return_value=_chunks("a.pdf"))):
        client.post("/pdf/ingest", files=[_pdf("a.pdf")], data={"subfolder": "2026"})

    assert mock_move_many.await_args.kwargs == {"folder": DEFAULT_COURSE_FOLDER, "subfolder": "2026"}


def test_ingest_invalid_folder_422_before_ingesting(client):
    store = client.app.state.vector_store
    store.add_chunks = AsyncMock()

    res = client.post("/pdf/ingest", files=[_pdf("a.pdf")], data={"folder": "   "})

    assert res.status_code == 422
    store.add_chunks.assert_not_awaited()


def test_ingest_placement_failure_keeps_ingestion_successful(client, monkeypatch):
    store = client.app.state.vector_store
    store.has_file.return_value = False
    store.add_chunks = AsyncMock(return_value=1)
    monkeypatch.setattr(
        routes.ingested_file_repository, "move_many", AsyncMock(side_effect=RuntimeError("db down"))
    )

    with patch("app.api.routes.extract_pdf_chunks", AsyncMock(return_value=_chunks("a.pdf"))):
        res = client.post("/pdf/ingest", files=[_pdf("a.pdf")], data={"folder": "SFI"})

    assert res.status_code == 200
    assert res.json()["status"] == "ok"
