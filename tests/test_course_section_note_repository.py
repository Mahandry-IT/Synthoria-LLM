import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.repositories import course_section_note_repository as repo

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

    row = await repo.upsert(session, SID, "0", "Revoir la partie sur les pertes.")

    session.add.assert_called_once_with(row)
    assert (row.session_id, row.section_id, row.note) == (SID, "0", "Revoir la partie sur les pertes.")
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_upsert_replaces_existing_note_without_adding_a_row():
    session = _session()
    from app.db.models import CourseSectionNote

    existing = CourseSectionNote(session_id=SID, section_id="0", note="ancienne note")
    session.get = AsyncMock(return_value=existing)

    row = await repo.upsert(session, SID, "0", "")  # note effacée

    assert row is existing and existing.note == ""
    session.add.assert_not_called()


@pytest.mark.asyncio
async def test_get_for_session_indexes_by_section_id():
    session = _session()
    from types import SimpleNamespace

    session.execute.return_value = MagicMock(
        scalars=lambda: MagicMock(all=lambda: [SimpleNamespace(section_id="0", note="a"), SimpleNamespace(section_id="2", note="b")])
    )

    assert await repo.get_for_session(session, SID) == {"0": "a", "2": "b"}
