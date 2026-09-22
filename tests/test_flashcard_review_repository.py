import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.db.models import FlashcardReview
from app.repositories import flashcard_review_repository as repo

SID = uuid.uuid4()
DUE = datetime(2026, 9, 25, tzinfo=timezone.utc)


def _session() -> MagicMock:
    session = MagicMock()
    session.commit = AsyncMock()
    session.get = AsyncMock(return_value=None)
    session.execute = AsyncMock()
    return session


@pytest.mark.asyncio
async def test_upsert_creates_missing_review_and_commits():
    session = _session()

    review = await repo.upsert(session, session_id=SID, card_id="1-0", box=0, due_at=DUE, last_result="correct")

    session.add.assert_called_once_with(review)
    session.commit.assert_awaited_once()
    assert (review.session_id, review.card_id, review.box, review.due_at, review.last_result) == (
        SID, "1-0", 0, DUE, "correct",
    )


@pytest.mark.asyncio
async def test_upsert_updates_existing_review_without_adding_a_row():
    session = _session()
    existing = FlashcardReview(session_id=SID, card_id="1-0", box=2, due_at=DUE, last_result="correct")
    session.get = AsyncMock(return_value=existing)

    review = await repo.upsert(session, session_id=SID, card_id="1-0", box=0, due_at=DUE, last_result="incorrect")

    assert review is existing and (existing.box, existing.last_result) == (0, "incorrect")
    session.add.assert_not_called()
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_for_sessions_indexes_by_session_and_card():
    session = _session()
    row = SimpleNamespace(session_id=SID, card_id="1-0")
    session.execute.return_value = MagicMock(scalars=lambda: MagicMock(all=lambda: [row]))

    assert await repo.get_for_sessions(session, [SID]) == {(SID, "1-0"): row}


@pytest.mark.asyncio
async def test_get_for_sessions_skips_query_without_sessions():
    session = _session()

    assert await repo.get_for_sessions(session, []) == {}
    session.execute.assert_not_awaited()
