"""Régénération d'un cours entier : plan reconstruit, génération rejouée, remplacement en place."""

import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import routes
from app.api.schemas import ApiPlannedSection
from app.core.config import Settings
from app.core.exceptions import GeminiInvalidResponseError, GeminiQuotaExceededError, GeminiUnavailableError
from app.db.models import CourseSectionNote
from app.repositories import course_session_repository
from app.schemas.course_generation import Section
from app.services import course_regenerator
from app.services.course_generator import INCOMPLETE_SECTION_NOTICE, _map_sections_to_course_sections
from app.services.course_plan_generator import _incomplete_section
from app.services.course_regenerator import is_incomplete_course, planned_sections_from_course, regenerate_course
from tests.test_course_plan_generator import _fake_gemini


def _stored_sections(*, with_introduction: bool = True, incomplete_title: str | None = "Beta") -> list[dict]:
    """Sections telles qu'enregistrées (passées par le vrai mapping d'API)."""
    sections = []
    if with_introduction:
        sections.append(Section.model_validate({
            "type": "introduction", "title": "Bienvenue", "blocks": [{"type": "text", "text": "intro"}], "subsections": [],
        }))
    sections.append(Section.model_validate({
        "type": "development", "title": "Alpha", "blocks": [],
        "subsections": [{"title": t, "blocks": [{"type": "text", "text": "x"}]} for t in ("Pourquoi", "Quoi", "Comment")],
    }))
    if incomplete_title:
        sections.append(_incomplete_section(ApiPlannedSection(type="development", title=incomplete_title, order=9)))
    return [s.model_dump(mode="json") for s in _map_sections_to_course_sections(sections)]


def _course(**overrides) -> dict:
    base = {
        "meta": {"title": "Graphes", "subject": "Algorithmique", "language": "fr", "depth": "standard"},
        "sections": _stored_sections(), "summary": "résumé", "quiz": [{"question": "?"}],
    }
    return {**base, **overrides}


def _row(course: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(), question="Explique les graphes", mode="file_question", filenames=["doc.pdf"],
        gemini_response=course or _course(),
    )


# ─── is_incomplete_course ──────────────────────────────────────


def test_course_with_an_incomplete_section_is_incomplete():
    assert is_incomplete_course(_course()) is True


@pytest.mark.parametrize("missing", [{"summary": ""}, {"summary": None}, {"quiz": None}, {"quiz": []}])
def test_course_delivered_without_its_wrap_up_is_incomplete(missing):
    complete = _course(sections=_stored_sections(incomplete_title=None))

    assert is_incomplete_course(complete) is False
    assert is_incomplete_course({**complete, **missing}) is True


# ─── planned_sections_from_course ──────────────────────────────


def test_plan_keeps_the_course_structure_titles_and_order():
    planned = planned_sections_from_course(_course())

    assert [(s.type, s.title) for s in planned] == [
        ("introduction", "Bienvenue"), ("development", "Alpha"), ("development", "Beta"),
        ("common_pitfalls", "Pièges courants"), ("summary", "Résumé"), ("next_steps", "Pour aller plus loin"),
    ]
    assert [s.order for s in planned] == [1, 2, 3, 4, 5, 6]


def test_plan_adds_an_introduction_to_a_course_that_lost_it():
    planned = planned_sections_from_course(_course(sections=_stored_sections(with_introduction=False)))

    assert [(s.type, s.title) for s in planned][:3] == [
        ("introduction", "Introduction"), ("development", "Alpha"), ("development", "Beta"),
    ]


def test_first_section_emptied_by_a_failed_regeneration_is_not_mistaken_for_the_introduction():
    emptied = {"id": "0", "title": "DFS", "quoi": "", "pourquoi": "", "comment": "", "subsections": [], "challenge": "Défi."}

    planned = planned_sections_from_course(_course(sections=[emptied, *_stored_sections(with_introduction=False)]))

    assert [(s.type, s.title) for s in planned][:2] == [("introduction", "Introduction"), ("development", "DFS")]


def test_legacy_first_section_with_flat_text_is_a_development_section():
    legacy = {"id": "0", "title": "Notion", "quoi": "q", "pourquoi": "p", "comment": "c"}

    planned = planned_sections_from_course(_course(sections=[legacy]))

    assert [(s.type, s.title) for s in planned][:2] == [("introduction", "Introduction"), ("development", "Notion")]


# ─── regenerate_course ─────────────────────────────────────────


@pytest.fixture
def retrieval(monkeypatch) -> AsyncMock:
    context = {"chunks": [], "file_sources": [], "search_query": "q", "web_research": None}
    build = AsyncMock(return_value=context)
    monkeypatch.setattr(course_regenerator, "build_retrieval_context", build)
    return build


@pytest.mark.asyncio
async def test_regenerated_course_keeps_structure_question_files_and_depth(retrieval):
    row = _row()

    result = await regenerate_course(row, _fake_gemini(), AsyncMock(), Settings(gemini_api_key="k"))

    assert [s.title for s in result.sections] == ["Bienvenue", "Alpha", "Beta"]
    assert not any(INCOMPLETE_SECTION_NOTICE in s.comment for s in result.sections)
    assert result.quiz and result.summary
    assert (result.meta.title, result.meta.depth) == ("Graphes", "standard")
    assert retrieval.await_args.args[0] == "Explique les graphes"
    assert retrieval.await_args.kwargs == {"mode": "file_question", "filename": ["doc.pdf"]}


@pytest.mark.asyncio
async def test_regenerating_a_course_without_any_section_is_refused(retrieval):
    with pytest.raises(ValueError):
        await regenerate_course(_row(_course(sections=[])), _fake_gemini(), AsyncMock(), Settings(gemini_api_key="k"))

    retrieval.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_wrap_up_propagates_so_that_nothing_is_replaced(retrieval):
    with pytest.raises(GeminiUnavailableError):
        await regenerate_course(_row(), _fake_gemini(fail_wrap_up=True), AsyncMock(), Settings(gemini_api_key="k"))


# ─── replace_course (repository) ───────────────────────────────


def _db_session(row, notes: list[CourseSectionNote]) -> MagicMock:
    session = MagicMock()
    session.get = AsyncMock(return_value=row)
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    notes_result = MagicMock()
    notes_result.scalars.return_value.all.return_value = notes
    session.execute = AsyncMock(return_value=notes_result)
    return session


@pytest.mark.asyncio
async def test_replace_course_moves_notes_to_the_section_with_the_same_title(retrieval, monkeypatch):
    # Ancien cours sans introduction : Alpha = "0", Beta = "1". Nouveau : Bienvenue "0", Alpha "1", Beta "2".
    row = _row(_course(sections=_stored_sections(with_introduction=False)))
    response = await regenerate_course(row, _fake_gemini(), AsyncMock(), Settings(gemini_api_key="k"))
    notes = [
        CourseSectionNote(session_id=row.id, section_id="0", note="note sur Alpha"),
        CourseSectionNote(session_id=row.id, section_id="1", note="note sur Beta"),
        CourseSectionNote(session_id=row.id, section_id="7", note="section disparue"),
    ]
    session = _db_session(row, notes)
    monkeypatch.setattr(course_session_repository, "get_by_id", AsyncMock(return_value=row))

    saved = await course_session_repository.replace_course(session, row.id, response)

    moved = {note.section_id: note.note for note in session.add_all.call_args.args[0]}
    assert moved == {"1": "note sur Alpha", "2": "note sur Beta"}
    assert [s["title"] for s in saved.gemini_response["sections"]] == ["Introduction", "Alpha", "Beta"]
    assert "session_id" not in saved.gemini_response
    deleted = [str(call.args[0]) for call in session.execute.await_args_list if str(call.args[0]).startswith("DELETE")]
    assert [sql.split()[2] for sql in deleted] == [
        "course_section_notes", "quiz_attempts", "quiz_questions", "flashcard_reviews", "flashcard_variants",
    ]
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_replace_course_on_unknown_session_changes_nothing(monkeypatch):
    session = _db_session(None, [])
    monkeypatch.setattr(course_session_repository, "get_by_id", AsyncMock(return_value=None))

    assert await course_session_repository.replace_course(session, uuid.uuid4(), MagicMock()) is None
    session.execute.assert_not_awaited()
    session.commit.assert_not_awaited()


# ─── POST /courses/{session_id}/regenerate ─────────────────────


@pytest.fixture
def env(monkeypatch):
    app = FastAPI()
    app.include_router(routes.router)

    @asynccontextmanager
    async def factory():
        yield object()

    app.state.db_session_factory = factory
    app.state.vector_store = AsyncMock()
    gemini = MagicMock()
    app.dependency_overrides[routes.get_gemini_client] = lambda: gemini

    row = _row()
    replace = AsyncMock(return_value=row)
    monkeypatch.setattr(routes.course_session_repository, "get_by_id", AsyncMock(return_value=row))
    monkeypatch.setattr(routes.course_session_repository, "replace_course", replace)
    return SimpleNamespace(client=TestClient(app), monkeypatch=monkeypatch, app=app, row=row, replace=replace)


def _url(session_id) -> str:
    return f"/courses/{session_id}/regenerate"


def test_route_replaces_the_course_in_place_and_returns_it(env, monkeypatch):
    import asyncio

    context = {"chunks": [], "file_sources": [], "search_query": "q", "web_research": None}
    monkeypatch.setattr(course_regenerator, "build_retrieval_context", AsyncMock(return_value=context))
    response = asyncio.run(regenerate_course(env.row, _fake_gemini(), AsyncMock(), Settings(gemini_api_key="k")))
    env.monkeypatch.setattr(routes, "regenerate_course", AsyncMock(return_value=response))

    res = env.client.post(_url(env.row.id))

    assert res.status_code == 200
    assert res.json()["session_id"] == str(env.row.id)
    assert [s["title"] for s in res.json()["sections"]] == ["Bienvenue", "Alpha", "Beta"]
    assert env.replace.await_args.args[1:] == (env.row.id, response)


@pytest.mark.parametrize("error, status_code", [
    (GeminiUnavailableError("down"), 503),
    (GeminiInvalidResponseError("clôture incomplète"), 502),
    (GeminiQuotaExceededError("quota"), 429),
])
def test_route_never_replaces_the_course_when_generation_fails(env, error, status_code):
    env.monkeypatch.setattr(routes, "regenerate_course", AsyncMock(side_effect=error))

    assert env.client.post(_url(env.row.id)).status_code == status_code
    env.replace.assert_not_awaited()


def test_route_refuses_a_complete_course(env):
    env.row.gemini_response = _course(sections=_stored_sections(incomplete_title=None))
    regenerate = AsyncMock()
    env.monkeypatch.setattr(routes, "regenerate_course", regenerate)

    assert env.client.post(_url(env.row.id)).status_code == 409
    regenerate.assert_not_awaited()


def test_route_unknown_session_is_404(env):
    env.monkeypatch.setattr(routes.course_session_repository, "get_by_id", AsyncMock(return_value=None))

    assert env.client.post(_url(uuid.uuid4())).status_code == 404


def test_route_shares_the_regenerate_rate_limit(env):
    env.app.dependency_overrides[routes.get_settings] = lambda: Settings(
        gemini_api_key="k", course_regenerate_rate_limit_per_minute=1
    )
    env.monkeypatch.setattr(routes, "regenerate_course", AsyncMock(side_effect=GeminiUnavailableError("down")))

    assert [env.client.post(_url(env.row.id)).status_code for _ in range(2)] == [503, 429]
