import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.dialects import postgresql

from app.db.models import CourseChatMessage
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
