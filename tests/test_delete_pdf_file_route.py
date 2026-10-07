from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import routes


@pytest.fixture
def client(monkeypatch):
    app = FastAPI()
    app.include_router(routes.router)
    app.state.vector_store = MagicMock()

    @asynccontextmanager
    async def factory():
        yield object()

    app.state.db_session_factory = factory
    monkeypatch.setattr(routes.ingested_file_repository, "delete", AsyncMock())
    return TestClient(app)


def test_delete_known_file_returns_204(client):
    client.app.state.vector_store.remove_file.return_value = 3

    res = client.delete("/pdf/files/doc.pdf")

    assert res.status_code == 204
    client.app.state.vector_store.remove_file.assert_called_once_with("doc.pdf")


def test_delete_unknown_file_returns_404(client):
    client.app.state.vector_store.remove_file.return_value = 0

    res = client.delete("/pdf/files/unknown.pdf")

    assert res.status_code == 404
