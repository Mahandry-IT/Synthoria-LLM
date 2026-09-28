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


def _row(sections: list[dict], next_steps: list[str] | None = None) -> SimpleNamespace:
    return SimpleNamespace(id=SID, gemini_response={"sections": sections, "next_steps": next_steps or []})


@pytest.mark.asyncio
async def test_appends_new_sections_after_existing_ones():
    row = _row([{"id": "0", "title": "A"}])
    session = _session(row)
    new_sections = [{"id": "1", "title": "B"}]

    result = await repo.append_sections(session, SID, new_sections, [])

    assert result.gemini_response["sections"] == [{"id": "0", "title": "A"}, {"id": "1", "title": "B"}]
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_replaces_next_steps_with_the_final_value():
    row = _row([], next_steps=["Ancienne piste"])
    session = _session(row)

    result = await repo.append_sections(session, SID, [], ["Nouvelle piste"])

    assert result.gemini_response["next_steps"] == ["Nouvelle piste"]


@pytest.mark.asyncio
async def test_strips_control_chars_from_the_new_sections():
    """Un titre/texte généré par le LLM peut recopier un caractère de contrôle (ex. NUL) depuis une
    source mal décodée : Postgres le refuse dans une colonne jsonb (UntranslatableCharacterError)."""
    row = _row([])
    session = _session(row)
    new_sections = [{"id": "1", "title": "Section\x00 corrompue"}]

    result = await repo.append_sections(session, SID, new_sections, [])

    assert result.gemini_response["sections"][0]["title"] == "Section corrompue"


@pytest.mark.asyncio
async def test_reassigns_a_new_dict_object_so_sqlalchemy_detects_the_change():
    row = _row([{"id": "0", "title": "A"}])
    original = row.gemini_response
    session = _session(row)

    await repo.append_sections(session, SID, [{"id": "1", "title": "B"}], [])

    assert row.gemini_response is not original  # mutation en place ne serait pas persistée


@pytest.mark.asyncio
async def test_returns_none_when_session_is_missing():
    session = _session(None)

    assert await repo.append_sections(session, SID, [{"id": "1"}], []) is None
    session.commit.assert_not_awaited()
