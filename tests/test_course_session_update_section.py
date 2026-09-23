import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.repositories import course_session_repository as repo

SID = uuid.uuid4()


def _session(row) -> MagicMock:
    session = MagicMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    session.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=lambda: row))
    return session


def _row(sections: list[dict]) -> SimpleNamespace:
    return SimpleNamespace(id=SID, gemini_response={"sections": sections})


@pytest.mark.asyncio
async def test_replaces_the_matching_section_by_id_reassigning_the_whole_dict():
    row = _row([{"id": "0", "title": "A"}, {"id": "1", "title": "B"}])
    session = _session(row)
    updated_section = {"id": "1", "title": "B régénérée"}

    result = await repo.update_section(session, SID, "1", updated_section)

    assert result.gemini_response["sections"] == [{"id": "0", "title": "A"}, updated_section]
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_reassigns_a_new_dict_object_so_sqlalchemy_detects_the_change():
    row = _row([{"id": "0", "title": "A"}])
    original = row.gemini_response
    session = _session(row)

    await repo.update_section(session, SID, "0", {"id": "0", "title": "A régénérée"})

    assert row.gemini_response is not original  # mutation en place ne serait pas persistée


@pytest.mark.asyncio
async def test_returns_none_when_session_is_missing():
    session = _session(None)

    assert await repo.update_section(session, SID, "0", {"id": "0"}) is None


@pytest.mark.asyncio
async def test_returns_none_when_section_id_is_not_found():
    row = _row([{"id": "0", "title": "A"}])
    session = _session(row)

    assert await repo.update_section(session, SID, "9", {"id": "9"}) is None
    session.commit.assert_not_awaited()
