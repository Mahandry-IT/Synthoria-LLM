import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import media_routes
from app.core.config import Settings, get_settings
from app.services.media.storage import storage_path_for


class FakeSessionFactory:
    def __call__(self):
        return self

    async def __aenter__(self):
        return object()

    async def __aexit__(self, *exc):
        return False


def _fake_asset(asset_id: uuid.UUID, sha256: str) -> SimpleNamespace:
    return SimpleNamespace(id=asset_id, sha256=sha256, mime="image/webp")


@pytest.fixture
def env(tmp_path, monkeypatch):
    settings = Settings(media_storage_dir=str(tmp_path))
    app = FastAPI()
    app.include_router(media_routes.router)
    app.state.db_session_factory = FakeSessionFactory()
    app.dependency_overrides[get_settings] = lambda: settings
    return SimpleNamespace(client=TestClient(app), app=app, settings=settings, tmp=tmp_path, monkeypatch=monkeypatch)


def test_get_media_asset_returns_404_when_unknown(env):
    env.monkeypatch.setattr(media_routes.media_asset_repository, "get", AsyncMock(return_value=None))
    resp = env.client.get(f"/media/{uuid.uuid4()}")
    assert resp.status_code == 404


def test_get_media_asset_rejects_non_uuid_path_param(env):
    resp = env.client.get("/media/not-a-uuid")
    assert resp.status_code == 422  # FastAPI valide asset_id comme UUID avant tout accès disque/DB


def test_get_media_asset_returns_410_when_file_missing_on_disk(env):
    asset_id = uuid.uuid4()
    asset = _fake_asset(asset_id, "a" * 64)
    env.monkeypatch.setattr(media_routes.media_asset_repository, "get", AsyncMock(return_value=asset))
    resp = env.client.get(f"/media/{asset_id}")
    assert resp.status_code == 410


def test_get_media_asset_serves_file_with_security_headers(env):
    asset_id = uuid.uuid4()
    sha256 = "b" * 64
    asset = _fake_asset(asset_id, sha256)
    env.monkeypatch.setattr(media_routes.media_asset_repository, "get", AsyncMock(return_value=asset))

    path = storage_path_for(env.settings, sha256)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-webp-bytes")

    resp = env.client.get(f"/media/{asset_id}")
    assert resp.status_code == 200
    assert resp.content == b"fake-webp-bytes"
    assert resp.headers["content-type"] == "image/webp"
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert resp.headers["cache-control"] == "public, max-age=31536000, immutable"


def test_get_media_asset_path_is_reconstructed_from_db_never_from_input(env):
    """`asset_id` ne détermine jamais le chemin disque : seul `sha256` (issu de la DB) le fait."""
    asset_id = uuid.uuid4()
    asset = _fake_asset(asset_id, "c" * 64)
    env.monkeypatch.setattr(media_routes.media_asset_repository, "get", AsyncMock(return_value=asset))
    resp = env.client.get(f"/media/{asset_id}")
    assert resp.status_code == 410  # le sha256 renvoyé par la DB n'a pas de fichier : jamais un chemin forgé depuis asset_id
