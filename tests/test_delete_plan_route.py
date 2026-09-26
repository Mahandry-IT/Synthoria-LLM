import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import routes


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(routes.router)

    @asynccontextmanager
    async def factory():
        yield object()

    app.state.db_session_factory = factory
    return TestClient(app)


def test_delete_known_plan_returns_204(client, monkeypatch):
    monkeypatch.setattr(routes.course_plan_repository, "delete", AsyncMock(return_value=True))

    res = client.delete(f"/courses/plans/{uuid.uuid4()}")

    assert res.status_code == 204


def test_delete_unknown_plan_returns_404(client, monkeypatch):
    monkeypatch.setattr(routes.course_plan_repository, "delete", AsyncMock(return_value=False))

    res = client.delete(f"/courses/plans/{uuid.uuid4()}")

    assert res.status_code == 404
