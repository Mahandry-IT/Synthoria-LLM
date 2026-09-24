import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.repositories import media_query_cache_repository as repo

NOW = datetime(2026, 9, 24, tzinfo=timezone.utc)


def _session() -> MagicMock:
    session = MagicMock()
    session.commit = AsyncMock()
    session.get = AsyncMock(return_value=None)
    session.execute = AsyncMock()
    return session


# ─── normalize_query_key ──────────────────────────────────────


def test_normalize_query_key_is_stable_across_case_and_whitespace():
    a = repo.normalize_query_key("  Chat  Noir ")
    b = repo.normalize_query_key("chat noir")
    assert a == b
    assert len(a) == 64  # hex sha256


def test_normalize_query_key_keeps_accents_distinct_from_unaccented():
    assert repo.normalize_query_key("électricité") != repo.normalize_query_key("electricite")


# ─── get_fresh / upsert / purge_older_than ───────────────────


@pytest.mark.asyncio
async def test_get_fresh_reports_not_found_when_missing():
    session = _session()
    hit = await repo.get_fresh(session, "key", timedelta(hours=1))
    assert hit.found is False


@pytest.mark.asyncio
async def test_get_fresh_reports_not_found_when_expired_without_deleting():
    session = _session()
    session.get = AsyncMock(return_value=SimpleNamespace(asset_id=uuid.uuid4(), created_at=NOW - timedelta(hours=2)))

    hit = await repo.get_fresh(session, "key", timedelta(hours=1))

    assert hit.found is False
    session.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_fresh_reports_positive_hit_within_ttl():
    asset_id = uuid.uuid4()
    session = _session()
    session.get = AsyncMock(return_value=SimpleNamespace(asset_id=asset_id, created_at=datetime.now(timezone.utc)))

    hit = await repo.get_fresh(session, "key", timedelta(hours=1))

    assert hit.found is True and hit.asset_id == asset_id


@pytest.mark.asyncio
async def test_get_fresh_reports_negative_hit_distinct_from_not_found():
    session = _session()
    session.get = AsyncMock(return_value=SimpleNamespace(asset_id=None, created_at=datetime.now(timezone.utc)))

    hit = await repo.get_fresh(session, "key", timedelta(hours=1))

    assert hit.found is True and hit.asset_id is None


@pytest.mark.asyncio
async def test_upsert_creates_missing_row_and_commits():
    session = _session()
    asset_id = uuid.uuid4()

    await repo.upsert(session, "key", "chat noir", asset_id)

    session.add.assert_called_once()
    added = session.add.call_args.args[0]
    assert (added.query_hash, added.query, added.asset_id) == ("key", "chat noir", asset_id)
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_upsert_creates_negative_entry_with_none_asset_id():
    session = _session()

    await repo.upsert(session, "key", "sujet introuvable", None)

    added = session.add.call_args.args[0]
    assert added.asset_id is None


@pytest.mark.asyncio
async def test_upsert_replaces_existing_row_without_adding():
    session = _session()
    new_asset_id = uuid.uuid4()
    existing = SimpleNamespace(query_hash="key", query="old", asset_id=None, created_at=NOW - timedelta(days=10))
    session.get = AsyncMock(return_value=existing)

    await repo.upsert(session, "key", "nouvelle requête", new_asset_id)

    session.add.assert_not_called()
    assert existing.query == "nouvelle requête" and existing.asset_id == new_asset_id
    assert existing.created_at > NOW - timedelta(days=1)
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_purge_older_than_deletes_and_commits_returning_row_count():
    session = _session()
    session.execute.return_value = MagicMock(rowcount=2)

    deleted = await repo.purge_older_than(session, timedelta(hours=168))

    assert deleted == 2
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_purge_older_than_handles_none_rowcount():
    session = _session()
    session.execute.return_value = MagicMock(rowcount=None)

    assert await repo.purge_older_than(session, timedelta(hours=168)) == 0
