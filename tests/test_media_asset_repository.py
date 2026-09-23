import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.db.models import MediaAsset
from app.repositories import media_asset_repository as repo


def _session() -> AsyncMock:
    session = AsyncMock()
    session.add = MagicMock()
    return session


def _asset_kwargs(**overrides) -> dict:
    kwargs = dict(
        kind="web", sha256="a" * 64, mime="image/webp", width=100, height=80,
        size_bytes=1234, storage_path="/data/media/aa/aaaa.webp", alt_text="Une image",
    )
    kwargs.update(overrides)
    return kwargs


@pytest.mark.asyncio
async def test_get_returns_session_get_result():
    session = _session()
    asset_id = uuid.uuid4()
    session.get = AsyncMock(return_value=MediaAsset(id=asset_id))
    result = await repo.get(session, asset_id)
    session.get.assert_awaited_once_with(MediaAsset, asset_id)
    assert result.id == asset_id


@pytest.mark.asyncio
async def test_create_persists_and_returns_asset():
    session = _session()
    asset = await repo.create(session, **_asset_kwargs())
    session.add.assert_called_once_with(asset)
    session.commit.assert_awaited_once()
    session.refresh.assert_awaited_once_with(asset)
    assert (asset.kind, asset.sha256, asset.mime, asset.width, asset.height) == ("web", "a" * 64, "image/webp", 100, 80)
    assert asset.bytes == 1234


@pytest.mark.asyncio
async def test_get_or_create_reuses_existing_asset_by_sha256():
    session = _session()
    existing = MediaAsset(id=uuid.uuid4(), sha256="b" * 64)
    result = MagicMock()
    result.scalar_one_or_none.return_value = existing
    session.execute = AsyncMock(return_value=result)

    kwargs = _asset_kwargs()
    kwargs.pop("sha256")
    asset = await repo.get_or_create(session, sha256="b" * 64, **kwargs)

    assert asset is existing
    session.add.assert_not_called()  # déduplication : pas de nouvelle ligne pour un sha256 déjà connu


@pytest.mark.asyncio
async def test_get_or_create_creates_when_sha256_unknown():
    session = _session()
    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    session.execute = AsyncMock(return_value=result)

    asset = await repo.get_or_create(session, **_asset_kwargs(sha256="c" * 64))

    session.add.assert_called_once()
    assert asset.sha256 == "c" * 64
