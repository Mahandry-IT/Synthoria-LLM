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


def test_failed_regeneration_never_overwrites_the_section(env):
    """La section reste telle quelle (donc toujours relançable) quand la régénération échoue."""
    env.monkeypatch.setattr(
        routes, "regenerate_section", AsyncMock(side_effect=GeminiInvalidResponseError("encore incomplète"))
    )
    update_section = AsyncMock()
    env.monkeypatch.setattr(routes.course_session_repository, "update_section", update_section)

    res = env.client.post(_regen_url(env.row.id))

    assert res.status_code == 502
    update_section.assert_not_awaited()


def test_section_left_empty_by_an_old_failed_regeneration_can_be_regenerated(env):
    env.row.gemini_response["sections"][0] = {
        "id": "0", "title": "Notion vidée", "incomplete": False, "quoi": "", "pourquoi": "", "comment": "",
        "challenge": "Un défi.", "subsections": [], "note": "", "worked_example": _WORKED_EXAMPLE,
    }
    env.monkeypatch.setattr(
        routes, "regenerate_section", AsyncMock(return_value=_good_regenerated_section("Notion vidée"))
    )
    env.monkeypatch.setattr(routes.course_session_repository, "update_section", AsyncMock(return_value=env.row))

    res = env.client.post(_regen_url(env.row.id))

    assert res.status_code == 200 and res.json()["subsections"]


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


def _history_env(monkeypatch, sections: list[dict], notes: dict[str, str], videos: list[dict] | None = None, video_notes: dict[str, str] | None = None):
    app = FastAPI()
    app.include_router(routes.router)

    @asynccontextmanager
    async def factory():
        yield object()

    app.state.db_session_factory = factory
    gemini_response = {
        "mode": "file_question", "format": "full_course", "introduction": {"quoi": "x"},
        "meta": {"title": "T", "subject": "S", "language": "fr", "generated_at": "2026-01-01T00:00:00Z"},
        "sources": [], "summary": "résumé", "sections": sections, "videos": videos or [],
    }
    row = SimpleNamespace(
        id=uuid.uuid4(), created_at=__import__("datetime").datetime(2026, 1, 1),
        question="Q ?", filenames=[], mode="file_question", gemini_response=gemini_response,
        folder="Général", subfolder="Non classé",
    )
    monkeypatch.setattr(routes.course_session_repository, "get_by_id", AsyncMock(return_value=row))
    monkeypatch.setattr(routes.course_section_note_repository, "get_for_session", AsyncMock(return_value=notes))
    monkeypatch.setattr(
        routes.course_video_note_repository, "get_for_session", AsyncMock(return_value=video_notes or {})
    )
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


def test_history_merges_notes_into_their_matching_videos(monkeypatch):
    videos = [{"video_id": "abc123", "url": "u", "embed_url": "e", "thumbnail_url": "t", "title": "V1", "channel": ""}]
    env = _history_env(monkeypatch, [], {}, videos=videos, video_notes={"abc123": "À revoir."})

    body = env.client.get(f"/courses/history/{env.row.id}").json()

    assert body["gemini_response"]["videos"][0]["note"] == "À revoir."


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


# ─── POST .../sections/regenerate-incomplete ──────────────────


def _bulk_url(session_id) -> str:
    return f"/courses/{session_id}/sections/regenerate-incomplete"


def _incomplete(section_id: str) -> dict:
    return {**INCOMPLETE, "id": section_id, "title": f"Notion {section_id}"}


def _bulk_env(env, outcomes: list, section_ids: list[str] | None = None):
    """`outcomes` : un résultat par section incomplète, dans l'ordre (exception à lever, ou None = succès)."""
    section_ids = section_ids or [str(i) for i in range(len(outcomes))]
    env.row.gemini_response = {"sections": [*(_incomplete(i) for i in section_ids), dict(HEALTHY, id="99")]}
    pending = list(outcomes)

    async def regenerate(row, title, *_args):
        outcome = pending.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return _good_regenerated_section(title)

    regenerate_mock = AsyncMock(side_effect=regenerate)
    update_section = AsyncMock(return_value=env.row)
    env.monkeypatch.setattr(routes, "regenerate_section", regenerate_mock)
    env.monkeypatch.setattr(routes.course_session_repository, "update_section", update_section)
    return regenerate_mock, update_section


def test_bulk_regenerates_every_incomplete_section_and_only_those(env):
    regenerate, update_section = _bulk_env(env, [None, None])

    res = env.client.post(_bulk_url(env.row.id))

    body = res.json()
    assert res.status_code == 200
    assert [s["id"] for s in body["regenerated"]] == ["0", "1"]
    assert body["failed"] == [] and body["remaining_section_ids"] == [] and body["stopped_reason"] is None
    assert [call.args[2] for call in update_section.await_args_list] == ["0", "1"]  # la section saine n'est pas touchée


def test_bulk_keeps_going_after_an_invalid_section(env):
    regenerate, update_section = _bulk_env(env, [GeminiInvalidResponseError("encore incomplète"), None])

    body = env.client.post(_bulk_url(env.row.id)).json()

    assert [s["id"] for s in body["regenerated"]] == ["1"]
    assert body["failed"] == [{"section_id": "0", "title": "Notion 0", "reason": "invalid_response"}]
    assert body["stopped_reason"] is None
    assert update_section.await_count == 1  # la section en échec n'est jamais réécrite


def test_bulk_stops_at_the_first_quota_error_and_reports_what_is_left(env):
    from app.core.exceptions import GeminiQuotaExceededError

    regenerate, _ = _bulk_env(env, [None, GeminiQuotaExceededError("quota"), None, None])

    res = env.client.post(_bulk_url(env.row.id))

    body = res.json()
    assert res.status_code == 200
    assert [s["id"] for s in body["regenerated"]] == ["0"]
    assert body["failed"] == [{"section_id": "1", "title": "Notion 1", "reason": "quota"}]
    assert body["remaining_section_ids"] == ["2", "3"] and body["stopped_reason"] == "quota"
    assert regenerate.await_count == 2  # aucun appel après l'erreur de quota


def test_bulk_stops_when_gemini_becomes_unavailable(env):
    _bulk_env(env, [None, GeminiUnavailableError("down"), None])

    body = env.client.post(_bulk_url(env.row.id)).json()

    assert body["stopped_reason"] == "unavailable" and body["remaining_section_ids"] == ["2"]


def test_bulk_with_nothing_done_reports_the_gemini_error_as_http(env):
    from app.core.exceptions import GeminiQuotaExceededError

    _, update_section = _bulk_env(env, [GeminiQuotaExceededError("quota"), None])

    res = env.client.post(_bulk_url(env.row.id))

    assert res.status_code == 429 and "Retry-After" in res.headers
    update_section.assert_not_awaited()

    _bulk_env(env, [GeminiUnavailableError("down")])
    assert env.client.post(_bulk_url(env.row.id)).status_code == 503


def test_bulk_is_capped_per_call(env):
    from app.core.config import Settings

    regenerate, _ = _bulk_env(env, [None, None, None])
    env.app.dependency_overrides[routes.get_settings] = lambda: Settings(
        gemini_api_key="k", course_regenerate_incomplete_max_sections=2
    )

    body = env.client.post(_bulk_url(env.row.id)).json()

    assert len(body["regenerated"]) == 2 and regenerate.await_count == 2
    assert body["remaining_section_ids"] == ["2"] and body["stopped_reason"] == "limit"


def test_bulk_without_incomplete_section_is_409(env):
    env.row.gemini_response = {"sections": [dict(HEALTHY)]}

    assert env.client.post(_bulk_url(env.row.id)).status_code == 409


def test_bulk_unknown_session_is_404(env):
    env.monkeypatch.setattr(routes.course_session_repository, "get_by_id", AsyncMock(return_value=None))

    assert env.client.post(_bulk_url(uuid.uuid4())).status_code == 404


def test_bulk_shares_the_regenerate_rate_limit(env):
    from app.core.config import Settings

    _bulk_env(env, [None] * 3, section_ids=["0", "1", "2"])
    env.app.dependency_overrides[routes.get_settings] = lambda: Settings(
        gemini_api_key="k", course_regenerate_rate_limit_per_minute=1, course_regenerate_incomplete_max_sections=1
    )

    assert env.client.post(_bulk_url(env.row.id)).status_code == 200
    assert env.client.post(_bulk_url(env.row.id)).status_code == 429


# ─── Une seule régénération groupée ou complète à la fois par cours ───


def test_section_and_bulk_regeneration_are_refused_while_the_course_is_being_regenerated(env):
    """Régression réelle : deux régénérations groupées lancées en parallèle refaisaient les mêmes
    sections (quota doublé), et le bouton par section répondait « pas incomplète » sur une section
    que la régénération groupée venait de réparer."""
    regenerate, _ = _bulk_env(env, [None])
    env.app.state.course_regenerations = {env.row.id}

    bulk = env.client.post(_bulk_url(env.row.id))
    single = env.client.post(_regen_url(env.row.id))

    assert bulk.status_code == 409 and single.status_code == 409
    assert "déjà en cours" in bulk.json()["detail"] and "déjà en cours" in single.json()["detail"]
    regenerate.assert_not_awaited()


def test_course_lock_is_released_after_success_and_after_failure(env):
    _bulk_env(env, [None])
    assert env.client.post(_bulk_url(env.row.id)).status_code == 200
    assert env.app.state.course_regenerations == set()

    _bulk_env(env, [GeminiUnavailableError("down")])
    assert env.client.post(_bulk_url(env.row.id)).status_code == 503
    assert env.app.state.course_regenerations == set()


def test_another_course_is_not_blocked(env):
    _bulk_env(env, [None])
    env.app.state.course_regenerations = {uuid.uuid4()}

    assert env.client.post(_bulk_url(env.row.id)).status_code == 200
