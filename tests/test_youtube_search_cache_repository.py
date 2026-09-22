from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.repositories import youtube_search_cache_repository as repo

NOW = datetime(2026, 9, 22, tzinfo=timezone.utc)


def _session() -> MagicMock:
    session = MagicMock()
    session.commit = AsyncMock()
    session.get = AsyncMock(return_value=None)
    session.execute = AsyncMock()
    return session


# ─── normalize_query_key ──────────────────────────────────────


def test_normalize_query_key_is_stable_across_case_and_whitespace():
    a = repo.normalize_query_key("  Cours  Thermodynamique ", language="fr", region="FR")
    b = repo.normalize_query_key("cours thermodynamique", language="fr", region="FR")

    assert a == b
    assert len(a) == 64  # hex sha256


def test_normalize_query_key_differs_by_language_or_region():
    base = repo.normalize_query_key("cours", language="fr", region="FR")

    assert repo.normalize_query_key("cours", language="en", region="FR") != base
    assert repo.normalize_query_key("cours", language="fr", region="US") != base


def test_normalize_query_key_keeps_accents_distinct_from_unaccented():
    assert repo.normalize_query_key("électricité", language="fr", region="FR") != repo.normalize_query_key(
        "electricite", language="fr", region="FR"
    )


# ─── get_fresh / upsert / purge_older_than ───────────────────


@pytest.mark.asyncio
async def test_get_fresh_returns_none_when_missing():
    session = _session()

    assert await repo.get_fresh(session, "key", timedelta(hours=1)) is None


@pytest.mark.asyncio
async def test_get_fresh_returns_none_when_expired_without_deleting():
    session = _session()
    session.get = AsyncMock(return_value=SimpleNamespace(candidates=[{"video_id": "x"}], fetched_at=NOW - timedelta(hours=2)))

    result = await repo.get_fresh(session, "key", timedelta(hours=1))

    assert result is None
    session.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_fresh_returns_candidates_when_within_ttl():
    session = _session()
    session.get = AsyncMock(
        return_value=SimpleNamespace(candidates=[{"video_id": "x"}], fetched_at=datetime.now(timezone.utc))
    )

    assert await repo.get_fresh(session, "key", timedelta(hours=1)) == [{"video_id": "x"}]


@pytest.mark.asyncio
async def test_upsert_creates_missing_row_and_commits():
    session = _session()

    await repo.upsert(session, "key", "cours ampli op", [{"video_id": "x"}])

    session.add.assert_called_once()
    added = session.add.call_args.args[0]
    assert (added.query_key, added.query, added.candidates) == ("key", "cours ampli op", [{"video_id": "x"}])
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_upsert_replaces_existing_row_without_adding():
    session = _session()
    existing = SimpleNamespace(query_key="key", query="old", candidates=[], fetched_at=NOW - timedelta(days=10))
    session.get = AsyncMock(return_value=existing)

    await repo.upsert(session, "key", "nouvelle requête", [{"video_id": "y"}])

    session.add.assert_not_called()
    assert existing.query == "nouvelle requête" and existing.candidates == [{"video_id": "y"}]
    assert existing.fetched_at > NOW - timedelta(days=1)
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_purge_older_than_deletes_and_commits_returning_row_count():
    session = _session()
    session.execute.return_value = MagicMock(rowcount=3)

    deleted = await repo.purge_older_than(session, timedelta(hours=72))

    assert deleted == 3
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_purge_older_than_handles_none_rowcount():
    session = _session()
    session.execute.return_value = MagicMock(rowcount=None)

    assert await repo.purge_older_than(session, timedelta(hours=72)) == 0
