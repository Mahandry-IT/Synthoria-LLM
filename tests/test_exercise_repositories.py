import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.dialects import postgresql

from app.db.models import ExerciseJob
from app.repositories import exercise_job_repository as jobs
from app.repositories import exercise_repository as exercises


def _session() -> AsyncMock:
    session = AsyncMock()
    session.add = MagicMock()
    session.add_all = MagicMock()
    return session


def _sql(statement) -> str:
    return str(statement.compile(dialect=postgresql.dialect()))


def test_claim_statement_skip_locked_pending_only_after_wait():
    sql = _sql(jobs.claim_statement())
    assert "FOR UPDATE SKIP LOCKED" in sql
    assert "exercise_jobs.status =" in sql
    assert "exercise_jobs.locked_at IS NULL" in sql and "exercise_jobs.locked_at <= now()" in sql
    assert "ORDER BY exercise_jobs.created_at" in sql


@pytest.mark.asyncio
async def test_claim_next_marks_generating_and_counts_attempt():
    session = _session()
    job = ExerciseJob(status="pending", attempts=1)
    result = MagicMock()
    result.scalar_one_or_none.return_value = job
    session.execute.return_value = result

    assert await jobs.claim_next(session) is job
    assert (job.status, job.attempts) == ("generating", 2)
    assert job.locked_at is not None
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_claim_next_empty_queue_rolls_back():
    session = _session()
    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    session.execute.return_value = result

    assert await jobs.claim_next(session) is None
    session.rollback.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_if_absent_is_insert_on_conflict_do_nothing():
    session = _session()
    result = MagicMock()
    result.scalar_one_or_none.return_value = uuid.uuid4()
    session.execute.return_value = result

    assert await jobs.create_if_absent(session, session_id=uuid.uuid4(), params_hash="h") is True
    sql = _sql(session.execute.await_args.args[0])
    assert "ON CONFLICT (session_id) DO NOTHING" in sql

    result.scalar_one_or_none.return_value = None  # conflit : la ligne existait déjà
    assert await jobs.create_if_absent(session, session_id=uuid.uuid4(), params_hash="h") is False


@pytest.mark.asyncio
async def test_restart_failed_only_touches_failed_jobs():
    session = _session()
    session.execute.return_value = MagicMock(rowcount=1)

    assert await jobs.restart_failed(session, uuid.uuid4(), params_hash="h") is True
    sql = _sql(session.execute.await_args.args[0])
    assert "exercise_jobs.status =" in sql and sql.startswith("UPDATE exercise_jobs")


@pytest.mark.asyncio
async def test_release_stale_counts_both_outcomes():
    session = _session()
    session.execute.side_effect = [MagicMock(rowcount=1), MagicMock(rowcount=2)]

    assert await jobs.release_stale(session, stale_minutes=10, max_attempts=3) == 3
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_requeue_sets_wait_and_refunds_attempt():
    session = _session()
    await jobs.requeue(session, uuid.uuid4(), error_message="x", refund_attempt=True, delay_seconds=30)
    stmt = session.execute.await_args.args[0]
    sql = _sql(stmt).lower()
    assert stmt.compile(dialect=postgresql.dialect()).params["status"] == "pending"
    assert "(now() +" in sql  # horloge de la base, pas celle du worker
    assert "greatest" in sql


@pytest.mark.asyncio
async def test_mark_done_guarded_on_generating_and_does_not_commit():
    session = _session()
    session.execute.return_value = MagicMock(rowcount=1)

    assert await jobs.mark_done(session, uuid.uuid4()) is True
    stmt = session.execute.await_args.args[0]
    assert "exercise_jobs.status =" in _sql(stmt)
    assert "generating" in stmt.compile(dialect=postgresql.dialect()).params.values()
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_mark_done_returns_false_and_rolls_back_when_job_is_no_longer_ours():
    session = _session()
    session.execute.return_value = MagicMock(rowcount=0)

    assert await jobs.mark_done(session, uuid.uuid4()) is False
    session.rollback.assert_awaited_once()
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_replace_all_does_not_commit_and_numbers_positions():
    session = _session()
    sid = uuid.uuid4()
    rows = [dict(section_id=None, title=f"T{i}", covers="c", statement=[], hints=[], duration_minutes=5,
                 expected_solution="s") for i in range(2)]

    created = await exercises.replace_all(session, sid, rows)

    assert [e.position for e in created] == [1, 2]
    session.commit.assert_not_awaited()
    assert _sql(session.execute.await_args.args[0]).startswith("DELETE FROM course_exercises")


@pytest.mark.asyncio
async def test_get_for_session_filters_on_both_ids():
    session = _session()
    session.execute.return_value = MagicMock(scalar_one_or_none=MagicMock(return_value=None))
    await exercises.get_for_session(session, uuid.uuid4(), uuid.uuid4())
    sql = _sql(session.execute.await_args.args[0])
    assert "course_exercises.id =" in sql and "course_exercises.session_id =" in sql
