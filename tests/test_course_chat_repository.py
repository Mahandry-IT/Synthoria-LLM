import importlib.util
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.dialects import postgresql

from app.db.models import CourseChatMessage
from app.db.schema_sync import ADDED_COLUMNS, DATA_BACKFILLS
from app.repositories import course_chat_repository as repo

SID = uuid.uuid4()
MIDNIGHT = datetime(2026, 10, 4, tzinfo=timezone.utc)


def _session() -> MagicMock:
    session = MagicMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    session.execute = AsyncMock()
    return session


def _sql(statement) -> str:
    return str(statement.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))


@pytest.mark.asyncio
async def test_count_user_messages_since_filters_session_role_and_day_start():
    session = _session()
    session.execute.return_value = MagicMock(scalar_one=lambda: 3)

    assert await repo.count_user_messages_since(session, SID, MIDNIGHT) == 3

    sql = _sql(session.execute.await_args.args[0])
    assert "count(*)" in sql
    assert f"course_chat_messages.session_id = '{SID}'" in sql
    assert "course_chat_messages.role = 'user'" in sql
    assert "course_chat_messages.created_at >= '2026-10-04 00:00:00+00:00'" in sql


@pytest.mark.asyncio
async def test_list_for_session_is_chronological():
    session = _session()
    session.execute.return_value = MagicMock(scalars=lambda: MagicMock(all=lambda: []))

    assert await repo.list_for_session(session, SID) == []

    sql = _sql(session.execute.await_args.args[0])
    assert "ORDER BY course_chat_messages.created_at, course_chat_messages.role DESC" in sql
    assert "course_chat_messages.deleted_at IS NULL" in sql


@pytest.mark.asyncio
async def test_count_user_messages_since_includes_deleted_messages():
    session = _session()
    session.execute.return_value = MagicMock(scalar_one=lambda: 0)

    await repo.count_user_messages_since(session, SID, MIDNIGHT)

    assert "deleted_at" not in _sql(session.execute.await_args.args[0])


@pytest.mark.asyncio
@pytest.mark.parametrize("for_update", [False, True])
async def test_get_active_message_filters_session_and_deleted(for_update):
    session = _session()
    session.execute.return_value = MagicMock(scalar_one_or_none=lambda: None)
    mid = uuid.uuid4()

    assert await repo.get_active_message(session, SID, mid, for_update=for_update) is None

    sql = _sql(session.execute.await_args.args[0])
    assert f"course_chat_messages.id = '{mid}'" in sql
    assert f"course_chat_messages.session_id = '{SID}'" in sql
    assert "course_chat_messages.deleted_at IS NULL" in sql
    assert ("FOR UPDATE" in sql) is for_update


@pytest.mark.asyncio
async def test_add_exchange_inserts_question_before_answer_in_one_commit():
    session = _session()
    session.flush = AsyncMock()
    calls: list[str] = []
    session.add.side_effect = lambda m: calls.append(f"add:{m.role}")
    session.flush.side_effect = lambda: calls.append("flush")
    user = CourseChatMessage(session_id=SID, role="user", content="q")
    assistant = CourseChatMessage(session_id=SID, role="assistant", content="r")

    assert await repo.add_exchange(session, user, assistant) == (user, assistant)

    assert calls == ["add:user", "flush", "add:assistant"]
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_soft_delete_branch_marks_message_and_descendants():
    session = _session()
    session.execute.return_value = MagicMock(rowcount=4)
    mid = uuid.uuid4()

    assert await repo.soft_delete_branch(session, SID, mid, MIDNIGHT) == 4

    sql = _sql(session.execute.await_args.args[0])
    assert "WITH RECURSIVE branch" in sql
    assert f"course_chat_messages.id = '{mid}'" in sql and f"course_chat_messages.session_id = '{SID}'" in sql
    assert "course_chat_messages.parent_id = branch.id" in sql
    assert "SET deleted_at='2026-10-04 00:00:00+00:00'" in sql
    assert "course_chat_messages.deleted_at IS NULL" in sql
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_soft_delete_all_marks_every_active_message_of_the_course():
    session = _session()
    session.execute.return_value = MagicMock(rowcount=6)

    assert await repo.soft_delete_all(session, SID, MIDNIGHT) == 6

    sql = _sql(session.execute.await_args.args[0])
    assert sql.startswith("UPDATE course_chat_messages SET deleted_at=")
    assert f"course_chat_messages.session_id = '{SID}'" in sql
    assert "course_chat_messages.deleted_at IS NULL" in sql
    session.commit.assert_awaited_once()


def test_messages_are_deleted_with_their_course():
    fk = next(iter(CourseChatMessage.__table__.c.session_id.foreign_keys))

    assert fk.column.table.name == "course_sessions" and fk.ondelete == "CASCADE"


def test_parent_is_a_nullable_self_reference_deleted_in_cascade():
    column = CourseChatMessage.__table__.c.parent_id
    fk = next(iter(column.foreign_keys))

    assert column.nullable is True
    assert fk.column.table.name == "course_chat_messages" and fk.ondelete == "CASCADE"
    assert CourseChatMessage.__table__.c.deleted_at.nullable is True


def _normalized(sql: str) -> str:
    return " ".join(sql.split())


def test_migration_017_matches_startup_schema_sync(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "migrations" / "versions" / "017_add_course_chat_message_versions.py"
    spec = importlib.util.spec_from_file_location("migration_017", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    executed: list[str] = []
    monkeypatch.setattr(module.op, "execute", executed.append, raising=False)

    module.upgrade()

    assert (module.revision, module.down_revision) == ("017", "016")
    startup = {_normalized(s) for s in (*ADDED_COLUMNS, *DATA_BACKFILLS)}
    assert all(_normalized(s) in startup for s in executed)
    assert any("ADD COLUMN IF NOT EXISTS parent_id" in s for s in executed)
    assert any("ADD COLUMN IF NOT EXISTS deleted_at" in s for s in executed)


def test_backfill_only_chains_sessions_never_chained_before():
    sql = _normalized(DATA_BACKFILLS[-1])

    assert "HAVING COUNT(*) > 1 AND COUNT(parent_id) = 0" in sql
    assert "LAG(id) OVER (PARTITION BY session_id ORDER BY created_at, role DESC)" in sql
