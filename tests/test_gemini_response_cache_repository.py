from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.repositories import gemini_response_cache_repository as repo

NOW = datetime(2026, 9, 30, tzinfo=timezone.utc)


def _session() -> MagicMock:
    session = MagicMock()
    session.commit = AsyncMock()
    session.get = AsyncMock(return_value=None)
    session.execute = AsyncMock()
    return session


# ─── compute_query_hash ────────────────────────────────────────


def test_compute_query_hash_is_stable():
    a = repo.compute_query_hash("rank_images", "prompt", "SchemaName", "hash1", "hash2")
    b = repo.compute_query_hash("rank_images", "prompt", "SchemaName", "hash1", "hash2")

    assert a == b
    assert len(a) == 64  # hex sha256


def test_compute_query_hash_differs_by_method_or_parts():
    base = repo.compute_query_hash("rank_images", "prompt", "hash1")

    assert repo.compute_query_hash("describe_images", "prompt", "hash1") != base
    assert repo.compute_query_hash("rank_images", "prompt", "hash2") != base


# ─── get_fresh / upsert / purge_older_than ─────────────────────


@pytest.mark.asyncio
async def test_get_fresh_returns_none_when_missing():
    session = _session()

    assert await repo.get_fresh(session, "key", timedelta(hours=1)) is None


@pytest.mark.asyncio
async def test_get_fresh_returns_none_when_expired_without_deleting():
    session = _session()
    session.get = AsyncMock(return_value=SimpleNamespace(response={"ok": True}, created_at=NOW - timedelta(hours=2)))

    result = await repo.get_fresh(session, "key", timedelta(hours=1))

    assert result is None
    session.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_fresh_returns_response_when_within_ttl():
    session = _session()
    session.get = AsyncMock(
        return_value=SimpleNamespace(response={"ok": True}, created_at=datetime.now(timezone.utc))
    )

    assert await repo.get_fresh(session, "key", timedelta(hours=1)) == {"ok": True}


@pytest.mark.asyncio
async def test_upsert_creates_missing_row_and_commits():
    session = _session()

    await repo.upsert(session, "key", "rank_images", {"best_index": 0})

    session.add.assert_called_once()
    added = session.add.call_args.args[0]
    assert (added.query_hash, added.method, added.response) == ("key", "rank_images", {"best_index": 0})
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_upsert_replaces_existing_row_without_adding():
    session = _session()
    existing = SimpleNamespace(query_hash="key", method="old", response={}, created_at=NOW - timedelta(days=10))
    session.get = AsyncMock(return_value=existing)

    await repo.upsert(session, "key", "describe_images", {"items": []})

    session.add.assert_not_called()
    assert existing.method == "describe_images" and existing.response == {"items": []}
    assert existing.created_at > NOW - timedelta(days=1)
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_purge_older_than_deletes_and_commits_returning_row_count():
    session = _session()
    session.execute.return_value = MagicMock(rowcount=3)

    deleted = await repo.purge_older_than(session, timedelta(hours=24))

    assert deleted == 3
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_purge_older_than_handles_none_rowcount():
    session = _session()
    session.execute.return_value = MagicMock(rowcount=None)

    assert await repo.purge_older_than(session, timedelta(hours=24)) == 0
