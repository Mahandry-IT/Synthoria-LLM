import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.db.models import CoursePlan
from app.repositories import course_plan_repository


def _session() -> AsyncMock:
    session = AsyncMock()
    session.add = MagicMock()  # Session.add est synchrone
    return session


@pytest.mark.asyncio
async def test_save_persists_pending_plan_and_commits():
    session = _session()
    expires_at = datetime.now(timezone.utc) + timedelta(hours=1)

    row = await course_plan_repository.save(
        session,
        question="Q",
        mode="file_question",
        filenames=["doc.pdf"],
        top_k=6,
        full_document=False,
        retrieval_context={"chunks": []},
        plan={"planned_sections": []},
        expires_at=expires_at,
    )

    session.add.assert_called_once_with(row)
    session.commit.assert_awaited_once()
    session.refresh.assert_awaited_once_with(row)
    assert isinstance(row, CoursePlan)
    assert (row.question, row.mode, row.filenames, row.expires_at) == ("Q", "file_question", ["doc.pdf"], expires_at)


@pytest.mark.asyncio
async def test_get_by_id_returns_row_or_none():
    session = _session()
    plan = CoursePlan(question="Q")
    result = MagicMock()
    result.scalar_one_or_none.return_value = plan
    session.execute.return_value = result

    assert await course_plan_repository.get_by_id(session, uuid.uuid4()) is plan

    result.scalar_one_or_none.return_value = None
    assert await course_plan_repository.get_by_id(session, uuid.uuid4()) is None


@pytest.mark.asyncio
async def test_mark_generated_updates_status():
    session = _session()
    plan = CoursePlan(question="Q", status="pending")
    result = MagicMock()
    result.scalar_one_or_none.return_value = plan
    session.execute.return_value = result

    await course_plan_repository.mark_generated(session, uuid.uuid4())

    assert plan.status == "generated"
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_mark_generated_is_noop_for_unknown_plan():
    session = _session()
    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    session.execute.return_value = result

    await course_plan_repository.mark_generated(session, uuid.uuid4())

    session.commit.assert_not_awaited()
