import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.repositories import course_video_note_repository as repo

SID = uuid.uuid4()


def _session() -> MagicMock:
    session = MagicMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    session.get = AsyncMock(return_value=None)
    session.execute = AsyncMock()
    return session


@pytest.mark.asyncio
async def test_upsert_creates_missing_row():
    session = _session()

    row = await repo.upsert(session, SID, "abc123", "Revoir cette vidéo.")

    session.add.assert_called_once_with(row)
    assert (row.session_id, row.video_id, row.note) == (SID, "abc123", "Revoir cette vidéo.")
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_upsert_replaces_existing_note_without_adding_a_row():
    session = _session()
    from app.db.models import CourseVideoNote

    existing = CourseVideoNote(session_id=SID, video_id="abc123", note="ancienne note")
    session.get = AsyncMock(return_value=existing)

    row = await repo.upsert(session, SID, "abc123", "")  # note effacée

    assert row is existing and existing.note == ""
    session.add.assert_not_called()


@pytest.mark.asyncio
async def test_get_for_session_indexes_by_video_id():
    session = _session()
    from types import SimpleNamespace

    session.execute.return_value = MagicMock(
        scalars=lambda: MagicMock(all=lambda: [SimpleNamespace(video_id="abc123", note="a"), SimpleNamespace(video_id="def456", note="b")])
    )

    assert await repo.get_for_session(session, SID) == {"abc123": "a", "def456": "b"}
