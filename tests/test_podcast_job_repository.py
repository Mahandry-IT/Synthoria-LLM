import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.dialects import postgresql

from app.db.models import PodcastJob
from app.repositories import podcast_job_repository as repo


def _session() -> AsyncMock:
    session = AsyncMock()
    session.add = MagicMock()
    return session


def test_claim_statement_uses_skip_locked_on_pending_oldest_first():
    sql = str(repo.claim_statement().compile(dialect=postgresql.dialect()))
    assert "FOR UPDATE SKIP LOCKED" in sql
    assert "podcast_jobs.status =" in sql
    assert "ORDER BY podcast_jobs.created_at" in sql
    assert "LIMIT" in sql


@pytest.mark.asyncio
async def test_create_persists_pending_job():
    session = _session()
    sid = uuid.uuid4()
    job = await repo.create(session, course_session_id=sid, params={"style": "x"}, params_hash="h" * 64)
    session.add.assert_called_once_with(job)
    session.commit.assert_awaited_once()
    assert (job.course_session_id, job.status, job.params_hash) == (sid, "pending", "h" * 64)


@pytest.mark.asyncio
async def test_claim_next_marks_scripting_and_counts_attempt():
    session = _session()
    job = PodcastJob(status="pending", attempts=1)
    result = MagicMock()
    result.scalar_one_or_none.return_value = job
    session.execute.return_value = result

    claimed = await repo.claim_next(session)

    assert claimed is job
    assert job.status == "scripting"
    assert job.attempts == 2
    assert job.locked_at is not None
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_claim_next_returns_none_when_queue_empty():
    session = _session()
    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    session.execute.return_value = result

    assert await repo.claim_next(session) is None
    session.rollback.assert_awaited_once()
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_release_stale_counts_both_updates():
    session = _session()
    failed, retried = MagicMock(rowcount=1), MagicMock(rowcount=2)
    session.execute.side_effect = [failed, retried]

    assert await repo.release_stale(session, stale_minutes=20, max_attempts=3) == 3
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_find_reusable_ignores_failed_jobs_in_query():
    session = _session()
    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    session.execute.return_value = result

    assert await repo.find_reusable(session, uuid.uuid4(), "h") is None
    stmt = str(session.execute.call_args.args[0].compile(dialect=postgresql.dialect()))
    assert "podcast_jobs.status !=" in stmt


@pytest.mark.asyncio
async def test_list_recent_joins_session_newest_first_with_limit():
    session = AsyncMock()
    job, course = MagicMock(), MagicMock()
    result = MagicMock()
    result.all.return_value = [(job, course)]
    session.execute.return_value = result

    rows = await repo.list_recent(session, 3)

    assert rows == [(job, course)]
    query = str(session.execute.await_args.args[0].compile(compile_kwargs={"literal_binds": True}))
    assert "JOIN course_sessions" in query
    assert "ORDER BY podcast_jobs.created_at DESC" in query
    assert "LIMIT 3" in query
