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
async def test_save_strips_control_chars_from_retrieval_context_and_plan():
    """Un chunk PDF ou une réponse Gemini peut contenir un \\x00 (police corrompue, texte source
    mal décodé recopié par le modèle) : Postgres le refuse dans une colonne jsonb
    (UntranslatableCharacterError), donc `save` doit le retirer avant l'insert."""
    session = _session()
    expires_at = datetime.now(timezone.utc) + timedelta(hours=1)

    row = await course_plan_repository.save(
        session,
        question="Q",
        mode="file_question",
        filenames=["doc.pdf"],
        top_k=6,
        full_document=False,
        retrieval_context={"chunks": [{"content": "Préface\x00 du livre", "metadata": {}}]},
        plan={"meta": {"title": "Titre\x00 corrompu"}},
        expires_at=expires_at,
    )

    assert row.retrieval_context["chunks"][0]["content"] == "Préface du livre"
    assert row.plan["meta"]["title"] == "Titre corrompu"


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


@pytest.mark.asyncio
async def test_list_pending_filters_pending_non_expired_newest_first_and_paginates():
    session = _session()
    plans = [MagicMock(), MagicMock()]
    count_result, rows_result = MagicMock(), MagicMock()
    count_result.scalar_one.return_value = 12
    rows_result.scalars.return_value.all.return_value = plans
    session.execute.side_effect = [count_result, rows_result]
    now = datetime.now(timezone.utc)

    rows, total = await course_plan_repository.list_pending(session, page=3, limit=5, now=now)

    assert (rows, total) == (plans, 12)
    query = str(session.execute.await_args_list[1].args[0].compile(compile_kwargs={"literal_binds": True}))
    assert "course_plans.status = 'pending'" in query
    assert "course_plans.expires_at >" in query
    assert "ORDER BY course_plans.created_at DESC" in query
    assert "LIMIT 5 OFFSET 10" in query
