from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.services.gemini_response_cache_manager import GeminiResponseCacheManager


def _settings(**overrides) -> Settings:
    return Settings(gemini_api_key="k", **overrides)


class _FakeSessionFactory:
    """Factory + context manager : `async with self._session_factory() as session`."""

    def __init__(self) -> None:
        self.session = object()

    def __call__(self):
        return self

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, *args):
        return False


@pytest.fixture
def repo(monkeypatch):
    fake_repo = SimpleNamespace(
        get_fresh=AsyncMock(return_value=None),
        upsert=AsyncMock(return_value=None),
    )
    monkeypatch.setattr("app.services.gemini_response_cache_manager.repo", fake_repo)
    return fake_repo


# ─── get ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_returns_none_without_session_factory():
    manager = GeminiResponseCacheManager(_settings(), session_factory=None)

    assert await manager.get("hash") is None


@pytest.mark.asyncio
async def test_get_returns_cached_response_within_ttl(repo):
    repo.get_fresh.return_value = {"query": "cached"}
    manager = GeminiResponseCacheManager(_settings(), _FakeSessionFactory())

    assert await manager.get("hash") == {"query": "cached"}


@pytest.mark.asyncio
async def test_get_returns_none_when_db_raises(repo):
    repo.get_fresh.side_effect = RuntimeError("db down")
    manager = GeminiResponseCacheManager(_settings(), _FakeSessionFactory())

    assert await manager.get("hash") is None


# ─── set ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_set_is_noop_without_session_factory():
    manager = GeminiResponseCacheManager(_settings(), session_factory=None)

    await manager.set("hash", "reformulate_query", {"query": "x"})


@pytest.mark.asyncio
async def test_set_upserts_via_repository(repo):
    manager = GeminiResponseCacheManager(_settings(), _FakeSessionFactory())

    await manager.set("hash", "reformulate_query", {"query": "x"})

    repo.upsert.assert_awaited_once()


@pytest.mark.asyncio
async def test_set_never_raises_when_db_unavailable(repo):
    repo.upsert.side_effect = RuntimeError("db down")
    manager = GeminiResponseCacheManager(_settings(), _FakeSessionFactory())

    await manager.set("hash", "reformulate_query", {"query": "x"})  # ne doit pas lever
