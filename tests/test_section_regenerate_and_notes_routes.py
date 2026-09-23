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
from app.services.course_generator import INCOMPLETE_SECTION_NOTICE

_WORKED_EXAMPLE = {"statement": "", "steps": [], "result": ""}
INCOMPLETE = {
    # incomplete=False à dessein : la détection ne doit JAMAIS se fier au seul champ stocké (une
    # session persistée avant l'existence de ce champ ne l'aurait pas), mais au texte de repli.
    "id": "0", "title": "Notion en échec", "incomplete": False,
    "quoi": "", "pourquoi": "", "comment": INCOMPLETE_SECTION_NOTICE, "note": "", "worked_example": _WORKED_EXAMPLE,
}
HEALTHY = {
    "id": "1", "title": "Notion en bonne santé", "incomplete": False,
    "quoi": "q", "pourquoi": "p", "comment": "c", "note": "", "worked_example": _WORKED_EXAMPLE,
}


def _good_regenerated_section(title: str) -> Section:
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
        gemini_response={"sections": [dict(INCOMPLETE), dict(HEALTHY)]},
    )
    monkeypatch.setattr(routes.course_session_repository, "get_by_id", AsyncMock(return_value=row))
    return SimpleNamespace(client=TestClient(app), gemini=gemini, monkeypatch=monkeypatch, app=app, row=row)


def _regen_url(session_id, section_id: str = "0") -> str:
    return f"/courses/{session_id}/sections/{section_id}/regenerate"


def _note_url(session_id, section_id: str = "0") -> str:
    return f"/courses/{session_id}/sections/{section_id}/note"


# ─── POST .../regenerate ──────────────────────────────────────


def test_regenerate_incomplete_section_succeeds(env):
    regenerate = AsyncMock(return_value=_good_regenerated_section("Notion en échec"))
    env.monkeypatch.setattr(routes, "regenerate_section", regenerate)
    update_section = AsyncMock(return_value=env.row)
    env.monkeypatch.setattr(routes.course_session_repository, "update_section", update_section)

    res = env.client.post(_regen_url(env.row.id))

    assert res.status_code == 200
    body = res.json()
    assert body["title"] == "Notion en échec" and body["incomplete"] is False and body["id"] == "0"
    regenerate.assert_awaited_once()
    update_section.assert_awaited_once()


def test_regenerate_unknown_session_is_404(env):
    env.monkeypatch.setattr(routes.course_session_repository, "get_by_id", AsyncMock(return_value=None))

    assert env.client.post(_regen_url(uuid.uuid4())).status_code == 404


def test_regenerate_unknown_section_is_404(env):
    assert env.client.post(_regen_url(env.row.id, "9")).status_code == 404


def test_regenerate_healthy_section_is_409(env):
    regenerate = AsyncMock()
    env.monkeypatch.setattr(routes, "regenerate_section", regenerate)

    res = env.client.post(_regen_url(env.row.id, "1"))

    assert res.status_code == 409
    regenerate.assert_not_awaited()


def test_regenerate_is_rate_limited(env):
    env.app.dependency_overrides[routes.get_settings] = lambda: SimpleNamespace(
        course_regenerate_rate_limit_per_minute=1, media_resolve_concurrency=4
    )
    env.monkeypatch.setattr(routes, "regenerate_section", AsyncMock(return_value=_good_regenerated_section("x")))
    env.monkeypatch.setattr(routes.course_session_repository, "update_section", AsyncMock(return_value=env.row))

    codes = [env.client.post(_regen_url(env.row.id)).status_code for _ in range(2)]

    assert codes == [200, 429]


def test_regenerate_gemini_error_is_translated_to_http(env):
    env.monkeypatch.setattr(routes, "regenerate_section", AsyncMock(side_effect=GeminiUnavailableError("down")))

    assert env.client.post(_regen_url(env.row.id)).status_code == 503


def test_regenerate_invalid_output_is_502(env):
    env.monkeypatch.setattr(routes, "regenerate_section", AsyncMock(side_effect=GeminiInvalidResponseError("bad")))

    assert env.client.post(_regen_url(env.row.id)).status_code == 502


def test_regenerate_section_vanishing_before_save_is_404(env):
    """Course course modifié entre la vérification et la sauvegarde (concurrence) : 404, pas une 500."""
    env.monkeypatch.setattr(routes, "regenerate_section", AsyncMock(return_value=_good_regenerated_section("x")))
    env.monkeypatch.setattr(routes.course_session_repository, "update_section", AsyncMock(return_value=None))

    assert env.client.post(_regen_url(env.row.id)).status_code == 404


# ─── PUT .../note ──────────────────────────────────────────────


def test_save_note_succeeds(env):
    saved = SimpleNamespace(note="Revoir les pertes fer.", updated_at=__import__("datetime").datetime(2026, 1, 1))
    upsert = AsyncMock(return_value=saved)
    env.monkeypatch.setattr(routes.course_section_note_repository, "upsert", upsert)

    res = env.client.put(_note_url(env.row.id, "1"), json={"note": "Revoir les pertes fer."})

    assert res.status_code == 200
    assert res.json()["note"] == "Revoir les pertes fer."
    upsert.assert_awaited_once()


def test_save_note_on_incomplete_section_is_also_allowed(env):
    """Aucune restriction : une section incomplète peut aussi recevoir une note."""
    saved = SimpleNamespace(note="x", updated_at=__import__("datetime").datetime(2026, 1, 1))
    env.monkeypatch.setattr(routes.course_section_note_repository, "upsert", AsyncMock(return_value=saved))

    assert env.client.put(_note_url(env.row.id, "0"), json={"note": "x"}).status_code == 200


def test_save_note_unknown_session_is_404(env):
    env.monkeypatch.setattr(routes.course_session_repository, "get_by_id", AsyncMock(return_value=None))

    assert env.client.put(_note_url(uuid.uuid4()), json={"note": "x"}).status_code == 404


def test_save_note_unknown_section_is_404(env):
    assert env.client.put(_note_url(env.row.id, "9"), json={"note": "x"}).status_code == 404


def test_save_empty_note_clears_it(env):
    saved = SimpleNamespace(note="", updated_at=__import__("datetime").datetime(2026, 1, 1))
    upsert = AsyncMock(return_value=saved)
    env.monkeypatch.setattr(routes.course_section_note_repository, "upsert", upsert)

    res = env.client.put(_note_url(env.row.id, "1"), json={"note": ""})

    assert res.status_code == 200 and res.json()["note"] == ""
    assert upsert.await_args.args[-1] == ""


def test_save_note_too_long_is_422(env):
    assert env.client.put(_note_url(env.row.id, "1"), json={"note": "x" * 2001}).status_code == 422


def test_save_note_is_rate_limited(env):
    env.app.dependency_overrides[routes.get_settings] = lambda: SimpleNamespace(course_note_rate_limit_per_minute=1)
    saved = SimpleNamespace(note="x", updated_at=__import__("datetime").datetime(2026, 1, 1))
    env.monkeypatch.setattr(routes.course_section_note_repository, "upsert", AsyncMock(return_value=saved))

    codes = [env.client.put(_note_url(env.row.id, "1"), json={"note": "x"}).status_code for _ in range(2)]

    assert codes == [200, 429]


# ─── GET /courses/history/{id} — fusion des notes ─────────────


def _history_env(monkeypatch, sections: list[dict], notes: dict[str, str]):
    app = FastAPI()
    app.include_router(routes.router)

    @asynccontextmanager
    async def factory():
        yield object()

    app.state.db_session_factory = factory
    gemini_response = {
        "mode": "file_question", "format": "full_course", "introduction": {"quoi": "x"},
        "meta": {"title": "T", "subject": "S", "language": "fr", "generated_at": "2026-01-01T00:00:00Z"},
        "sources": [], "summary": "résumé", "sections": sections,
    }
    row = SimpleNamespace(
        id=uuid.uuid4(), created_at=__import__("datetime").datetime(2026, 1, 1),
        question="Q ?", filenames=[], mode="file_question", gemini_response=gemini_response,
    )
    monkeypatch.setattr(routes.course_session_repository, "get_by_id", AsyncMock(return_value=row))
    monkeypatch.setattr(routes.course_section_note_repository, "get_for_session", AsyncMock(return_value=notes))
    return SimpleNamespace(client=TestClient(app), row=row)


def test_history_merges_notes_into_their_matching_sections(monkeypatch):
    env = _history_env(monkeypatch, [dict(INCOMPLETE), dict(HEALTHY)], {"1": "Revoir cette partie."})

    body = env.client.get(f"/courses/history/{env.row.id}").json()

    sections = body["gemini_response"]["sections"]
    assert sections[0]["note"] == "" and sections[1]["note"] == "Revoir cette partie."


def test_history_without_notes_leaves_sections_untouched(monkeypatch):
    env = _history_env(monkeypatch, [dict(HEALTHY)], {})

    body = env.client.get(f"/courses/history/{env.row.id}").json()

    assert body["gemini_response"]["sections"][0]["note"] == ""


def test_history_recomputes_incomplete_for_a_session_predating_the_field(monkeypatch):
    """Reproduction du bug signalé : une section historique avec `incomplete` absent/faux, mais
    dont le contenu est le texte de repli, doit ressortir `incomplete: true` à la lecture."""
    legacy_section = {
        "id": "0", "title": "Notion en échec", "quoi": "", "pourquoi": "", "note": "",
        "comment": INCOMPLETE_SECTION_NOTICE, "worked_example": _WORKED_EXAMPLE,
        # pas de clé "incomplete" du tout : session persistée avant l'existence de ce champ
    }
    env = _history_env(monkeypatch, [legacy_section], {})

    body = env.client.get(f"/courses/history/{env.row.id}").json()

    assert body["gemini_response"]["sections"][0]["incomplete"] is True
