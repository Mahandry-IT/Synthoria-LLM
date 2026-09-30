from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.services.gemini_quota_manager import GeminiQuotaManager


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
        get_states=AsyncMock(return_value={}),
        increment=AsyncMock(return_value=1),
        mark_exhausted=AsyncMock(return_value=None),
    )
    monkeypatch.setattr("app.services.gemini_quota_manager.repo", fake_repo)
    return fake_repo


# ─── is_available ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_is_available_true_without_session_factory():
    manager = GeminiQuotaManager(_settings(), session_factory=None)

    assert await manager.is_available("flash-lite") is True


@pytest.mark.asyncio
async def test_is_available_true_when_model_never_seen(repo):
    manager = GeminiQuotaManager(_settings(), _FakeSessionFactory())

    assert await manager.is_available("flash-lite") is True


@pytest.mark.asyncio
async def test_is_available_false_when_exhausted_until_in_future(repo):
    repo.get_states.return_value = {
        "flash-lite": SimpleNamespace(
            day=date.today(), request_count=1, exhausted_until=datetime.now(timezone.utc) + timedelta(hours=1)
        )
    }
    manager = GeminiQuotaManager(_settings(), _FakeSessionFactory())

    assert await manager.is_available("flash-lite") is False


@pytest.mark.asyncio
async def test_is_available_true_when_exhausted_until_in_past(repo):
    repo.get_states.return_value = {
        "flash-lite": SimpleNamespace(
            day=date.today(), request_count=1, exhausted_until=datetime.now(timezone.utc) - timedelta(hours=1)
        )
    }
    manager = GeminiQuotaManager(_settings(), _FakeSessionFactory())

    assert await manager.is_available("flash-lite") is True


@pytest.mark.asyncio
async def test_is_available_false_when_rpd_budget_reached(repo):
    repo.get_states.return_value = {
        "flash-lite": SimpleNamespace(day=date.today(), request_count=100, exhausted_until=None)
    }
    manager = GeminiQuotaManager(
        _settings(gemini_model_rpd_limits={"flash-lite": 100}), _FakeSessionFactory()
    )

    assert await manager.is_available("flash-lite") is False


@pytest.mark.asyncio
async def test_is_available_ignores_rpd_count_from_a_stale_day(repo):
    repo.get_states.return_value = {
        "flash-lite": SimpleNamespace(day=date(2020, 1, 1), request_count=999, exhausted_until=None)
    }
    manager = GeminiQuotaManager(
        _settings(gemini_model_rpd_limits={"flash-lite": 10}), _FakeSessionFactory()
    )

    assert await manager.is_available("flash-lite") is True


@pytest.mark.asyncio
async def test_is_available_uses_cache_within_ttl_without_db_call(repo):
    manager = GeminiQuotaManager(_settings(gemini_quota_cache_ttl_seconds=60.0), _FakeSessionFactory())

    await manager.is_available("flash-lite")
    await manager.is_available("flash-lite")

    assert repo.get_states.await_count == 1


@pytest.mark.asyncio
async def test_is_available_refreshes_after_ttl_expires(repo, monkeypatch):
    # Rebind le nom `time` DANS le module (pas le module `time` partagé, dont dépend asyncio) :
    # muter `time.monotonic` globalement casserait l'event loop lui-même.
    clock = iter([100.0, 200.0, 200.0])  # 1er appel (write), 2e appel (check TTL expiré, write)
    monkeypatch.setattr(
        "app.services.gemini_quota_manager.time", SimpleNamespace(monotonic=lambda: next(clock))
    )
    manager = GeminiQuotaManager(_settings(gemini_quota_cache_ttl_seconds=1.0), _FakeSessionFactory())

    await manager.is_available("flash-lite")
    await manager.is_available("flash-lite")

    assert repo.get_states.await_count == 2


@pytest.mark.asyncio
async def test_is_available_defaults_true_when_db_raises(repo):
    repo.get_states.side_effect = RuntimeError("db down")
    manager = GeminiQuotaManager(_settings(), _FakeSessionFactory())

    assert await manager.is_available("flash-lite") is True


# ─── record_success / record_daily_exhausted ───────────────────


@pytest.mark.asyncio
async def test_record_success_increments_via_repository(repo):
    manager = GeminiQuotaManager(_settings(), _FakeSessionFactory())

    await manager.record_success("flash-lite")

    repo.increment.assert_awaited_once()


@pytest.mark.asyncio
async def test_record_success_never_raises_when_db_unavailable(repo):
    repo.increment.side_effect = RuntimeError("db down")
    manager = GeminiQuotaManager(_settings(), _FakeSessionFactory())

    await manager.record_success("flash-lite")  # ne doit pas lever


@pytest.mark.asyncio
async def test_record_success_is_noop_without_session_factory(repo):
    manager = GeminiQuotaManager(_settings(), session_factory=None)

    await manager.record_success("flash-lite")

    repo.increment.assert_not_awaited()


@pytest.mark.asyncio
async def test_record_daily_exhausted_marks_until_next_pacific_midnight_and_updates_cache(repo):
    manager = GeminiQuotaManager(_settings(), _FakeSessionFactory())

    await manager.record_daily_exhausted("flash-lite")

    repo.mark_exhausted.assert_awaited_once()
    assert await manager.is_available("flash-lite") is False  # cache local mis à jour immédiatement


@pytest.mark.asyncio
async def test_record_daily_exhausted_never_raises_when_db_unavailable(repo):
    repo.mark_exhausted.side_effect = RuntimeError("db down")
    manager = GeminiQuotaManager(_settings(), _FakeSessionFactory())

    await manager.record_daily_exhausted("flash-lite")  # ne doit pas lever
