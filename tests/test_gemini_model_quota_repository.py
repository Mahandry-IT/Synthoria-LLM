from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.exc import IntegrityError

from app.repositories import gemini_model_quota_repository as repo

TODAY = date(2026, 9, 30)


def _session() -> MagicMock:
    session = MagicMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.get = AsyncMock(return_value=None)
    session.execute = AsyncMock()
    return session


# ─── increment ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_increment_returns_new_count_when_row_updated():
    """La ligne existe déjà pour aujourd'hui (ou est réinitialisée par le CASE SQL, indifférent
    du point de vue Python) : l'UPDATE RETURNING renvoie directement le nouveau compteur."""
    session = _session()
    session.execute.return_value = MagicMock(first=lambda: (5,))

    result = await repo.increment(session, "gemini-3.5-flash-lite", TODAY)

    assert result == 5
    session.commit.assert_awaited_once()
    session.add.assert_not_called()


@pytest.mark.asyncio
async def test_increment_inserts_row_when_model_never_seen():
    session = _session()
    session.execute.return_value = MagicMock(first=lambda: None)  # 0 ligne affectée

    result = await repo.increment(session, "gemini-3.5-flash-lite", TODAY)

    assert result == 1
    session.add.assert_called_once()
    added = session.add.call_args.args[0]
    assert (added.model, added.day, added.request_count) == ("gemini-3.5-flash-lite", TODAY, 1)
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_increment_falls_back_to_update_on_concurrent_insert_race():
    """Un autre process (api/worker) insère la ligne entre l'UPDATE et l'INSERT : l'INSERT lève
    IntegrityError (PK), on doit retomber sur le même UPDATE plutôt que propager l'erreur."""
    session = _session()
    session.execute.side_effect = [
        MagicMock(first=lambda: None),  # 1er UPDATE : rien à mettre à jour
        MagicMock(scalar_one=lambda: 1),  # 2e UPDATE (repli) : la ligne existe maintenant
    ]
    session.commit.side_effect = [IntegrityError("insert", {}, Exception("dup")), None]

    result = await repo.increment(session, "gemini-3.5-flash-lite", TODAY)

    assert result == 1
    session.rollback.assert_awaited_once()
    assert session.execute.await_count == 2


# ─── mark_exhausted ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_mark_exhausted_creates_row_when_absent():
    session = _session()
    until = datetime(2026, 10, 1, tzinfo=timezone.utc)

    await repo.mark_exhausted(session, "gemini-3.6-flash", until)

    session.add.assert_called_once()
    added = session.add.call_args.args[0]
    assert (added.model, added.exhausted_until) == ("gemini-3.6-flash", until)
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_mark_exhausted_updates_existing_row_without_adding():
    session = _session()
    existing = SimpleNamespace(model="gemini-3.6-flash", exhausted_until=None, updated_at=None)
    session.get = AsyncMock(return_value=existing)
    until = datetime(2026, 10, 1, tzinfo=timezone.utc)

    await repo.mark_exhausted(session, "gemini-3.6-flash", until)

    session.add.assert_not_called()
    assert existing.exhausted_until == until
    session.commit.assert_awaited_once()


# ─── get_states ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_states_returns_empty_dict_without_querying_for_empty_list():
    session = _session()

    assert await repo.get_states(session, [], TODAY) == {}
    session.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_states_returns_dict_keyed_by_model():
    session = _session()
    row_a = SimpleNamespace(model="a")
    row_b = SimpleNamespace(model="b")
    session.execute.return_value = MagicMock(scalars=lambda: MagicMock(all=lambda: [row_a, row_b]))

    result = await repo.get_states(session, ["a", "b"], TODAY)

    assert result == {"a": row_a, "b": row_b}
