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


@pytest.mark.asyncio
async def test_add_exchange_commits_both_messages_once():
    session = _session()
    user = CourseChatMessage(session_id=SID, role="user", content="q")
    assistant = CourseChatMessage(session_id=SID, role="assistant", content="r")

    assert await repo.add_exchange(session, user, assistant) == (user, assistant)

    session.add_all.assert_called_once_with([user, assistant])
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
