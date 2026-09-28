import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import routes
from app.core.exceptions import GeminiInvalidResponseError, GeminiUnavailableError
from app.schemas.course_generation import Section

EXISTING = {
    "id": "0", "title": "Introduction", "incomplete": False,
    "quoi": "q", "pourquoi": "p", "comment": "c", "note": "",
    "worked_example": {"statement": "", "steps": [], "result": ""},
}


def _good_new_section(title: str = "Nouvelle notion") -> Section:
    return Section.model_validate({
        "type": "development", "title": title, "blocks": [],
        "subsections": [
            {"title": "Pourquoi", "blocks": [{"type": "text", "text": "raison"}]},
            {"title": "Quoi", "blocks": [{"type": "text", "text": "définition"}]},
            {"title": "Comment", "blocks": [{"type": "text", "text": "mécanisme"}]},
        ],
    })


@pytest.fixture
def env(monkeypatch):
    app = FastAPI()
    app.include_router(routes.router)

    @asynccontextmanager
    async def factory():
        yield object()

    app.state.db_session_factory = factory
    app.state.vector_store = AsyncMock()
    gemini = AsyncMock()
    app.dependency_overrides[routes.get_gemini_client] = lambda: gemini

    row = SimpleNamespace(
        id=uuid.uuid4(), question="Q ?", mode="file_question", filenames=[],
        gemini_response={"sections": [dict(EXISTING)], "next_steps": ["Piste A"]},
    )
    monkeypatch.setattr(routes.course_session_repository, "get_by_id", AsyncMock(return_value=row))
    return SimpleNamespace(client=TestClient(app), gemini=gemini, monkeypatch=monkeypatch, app=app, row=row)


def _url(session_id) -> str:
    return f"/courses/{session_id}/sections"


def test_add_sections_succeeds_and_returns_the_final_next_steps(env):
    env.monkeypatch.setattr(
        routes, "add_course_sections", AsyncMock(return_value=([_good_new_section("Nouvelle notion")], []))
    )
    env.monkeypatch.setattr(routes.course_session_repository, "append_sections", AsyncMock(return_value=env.row))

    res = env.client.post(_url(env.row.id), json={"instructions": "Parle des pertes fer"})

    assert res.status_code == 200
    body = res.json()
    assert body["sections"][0]["title"] == "Nouvelle notion"
    # Se poursuit après la section existante (id="0")
    assert body["sections"][0]["id"] == "1"
    assert body["next_steps"] == []


def test_add_sections_with_empty_body_defaults_to_empty_instructions(env):
    add = AsyncMock(return_value=([_good_new_section()], ["Piste A"]))
    env.monkeypatch.setattr(routes, "add_course_sections", add)
    env.monkeypatch.setattr(routes.course_session_repository, "append_sections", AsyncMock(return_value=env.row))

    res = env.client.post(_url(env.row.id), json={})

    assert res.status_code == 200
    assert add.await_args.args[1] == ""


def test_add_sections_unknown_session_is_404(env):
    env.monkeypatch.setattr(routes.course_session_repository, "get_by_id", AsyncMock(return_value=None))

    assert env.client.post(_url(uuid.uuid4()), json={}).status_code == 404


def test_add_sections_is_rate_limited(env):
    env.app.dependency_overrides[routes.get_settings] = lambda: SimpleNamespace(
        add_course_sections_rate_limit_per_minute=1, media_resolve_concurrency=4
    )
    env.monkeypatch.setattr(routes, "add_course_sections", AsyncMock(return_value=([_good_new_section()], [])))
    env.monkeypatch.setattr(routes.course_session_repository, "append_sections", AsyncMock(return_value=env.row))

    codes = [env.client.post(_url(env.row.id), json={}).status_code for _ in range(2)]

    assert codes == [200, 429]


def test_add_sections_gemini_error_is_translated_to_http(env):
    env.monkeypatch.setattr(routes, "add_course_sections", AsyncMock(side_effect=GeminiUnavailableError("down")))

    assert env.client.post(_url(env.row.id), json={}).status_code == 503


def test_add_sections_invalid_output_is_502(env):
    env.monkeypatch.setattr(routes, "add_course_sections", AsyncMock(side_effect=GeminiInvalidResponseError("bad")))

    assert env.client.post(_url(env.row.id), json={}).status_code == 502


def test_add_sections_empty_result_is_502(env):
    """`add_course_sections` lève déjà si le modèle ne renvoie rien, mais un mapping vide (aucune
    section exploitable après filtrage) doit aussi rester un échec explicite, pas une réponse vide."""
    env.monkeypatch.setattr(routes, "add_course_sections", AsyncMock(return_value=([], [])))

    assert env.client.post(_url(env.row.id), json={}).status_code == 502


def test_add_sections_session_vanishing_before_save_is_404(env):
    """Cours supprimé entre la génération et la sauvegarde (concurrence) : 404, pas une 500."""
    env.monkeypatch.setattr(routes, "add_course_sections", AsyncMock(return_value=([_good_new_section()], [])))
    env.monkeypatch.setattr(routes.course_session_repository, "append_sections", AsyncMock(return_value=None))

    assert env.client.post(_url(env.row.id), json={}).status_code == 404


def test_add_sections_instructions_too_long_is_422(env):
    assert env.client.post(_url(env.row.id), json={"instructions": "x" * 1001}).status_code == 422
