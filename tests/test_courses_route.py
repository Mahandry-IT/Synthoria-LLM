from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.exceptions import (
    GeminiInvalidResponseError,
    GeminiQuotaExceededError,
    GeminiUnavailableError,
)
from app.main import app
from app.services.course_plan_generator import MoreSectionsResult

VALID_RESPONSE = {
    "mode": "file_question",
    "format": "focused_answer",
    "meta": {"title": "T", "subject": "S", "language": "fr", "generated_at": "2026-08-18T10:00:00Z"},
    "sources": [{"type": "web", "label": "W", "reference": "https://w"}],
    "answer": {
        "quoi": "quoi",
        "pourquoi": "pourquoi",
        "comment": "comment",
        "worked_example": {"statement": "s", "steps": [{"id": "1", "content": "e1"}], "result": "r"},
        "key_points": [],
    },
    "summary": "résumé",
    "next_steps": [],
}


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        app.state.ollama_client = AsyncMock()
        app.state.vector_store = AsyncMock()
        app.state.gemini_client = AsyncMock()
        yield test_client


def test_generate_course_success(client):
    with patch(
        "app.api.routes.generate_course_from_question", new_callable=AsyncMock
    ) as mock_generate:
        from app.api.schemas import CourseGenerationResponse

        mock_generate.return_value = CourseGenerationResponse.model_validate(VALID_RESPONSE)

        res = client.post("/courses/generate", json={"question": "Comment fonctionne X ?"})

        assert res.status_code == 200
        assert res.json()["mode"] == "file_question"


@pytest.mark.parametrize("payload", [
    {"question": "   "},
    {},
    {"question": None},
])
def test_generate_course_empty_question_uses_default(client, payload):
    with patch(
        "app.api.routes.generate_course_from_question", new_callable=AsyncMock
    ) as mock_generate:
        from app.api.schemas import COURSE_DEFAULT_QUESTION, CourseGenerationResponse

        mock_generate.return_value = CourseGenerationResponse.model_validate(VALID_RESPONSE)

        res = client.post("/courses/generate", json=payload)

        assert res.status_code == 200
        mock_generate.assert_called_once()
        assert mock_generate.call_args.kwargs["question"] == COURSE_DEFAULT_QUESTION


def test_generate_course_question_too_long_rejected(client):
    res = client.post("/courses/generate", json={"question": "a" * 3000})
    assert res.status_code == 413


def test_generate_course_gemini_unavailable(client):
    with patch(
        "app.api.routes.generate_course_from_question", new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.side_effect = GeminiUnavailableError("down")
        res = client.post("/courses/generate", json={"question": "question"})
        assert res.status_code == 503


def test_generate_course_gemini_quota_exceeded(client):
    with patch(
        "app.api.routes.generate_course_from_question", new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.side_effect = GeminiQuotaExceededError("quota")
        res = client.post("/courses/generate", json={"question": "question"})
        assert res.status_code == 429


def test_generate_course_gemini_invalid_response(client):
    with patch(
        "app.api.routes.generate_course_from_question", new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.side_effect = GeminiInvalidResponseError("bad json")
        res = client.post("/courses/generate", json={"question": "question"})
        assert res.status_code == 502


# --- Mode 3 : question_only (question seule + recherche web) ---


VALID_QUESTION_ONLY_RESPONSE = {
    **VALID_RESPONSE,
    "mode": "question_only",
    "sources": [{"type": "web", "label": "W", "reference": "https://w"}],
}


def test_generate_course_question_only_auto_detect(client):
    """Sans filename ni mode explicite → auto-détection en question_only."""
    with patch(
        "app.api.routes.generate_course_from_question", new_callable=AsyncMock
    ) as mock_generate:
        from app.api.schemas import CourseGenerationResponse

        mock_generate.return_value = CourseGenerationResponse.model_validate(
            VALID_QUESTION_ONLY_RESPONSE
        )

        res = client.post("/courses/generate", json={"question": "Qu'est-ce que l'IA ?"})

        assert res.status_code == 200
        assert res.json()["mode"] == "question_only"
        mock_generate.assert_called_once()
        assert mock_generate.call_args.kwargs["mode"] == "question_only"


def test_generate_course_question_only_explicit_mode(client):
    """Mode question_only forcé explicitement."""
    with patch(
        "app.api.routes.generate_course_from_question", new_callable=AsyncMock
    ) as mock_generate:
        from app.api.schemas import CourseGenerationResponse

        mock_generate.return_value = CourseGenerationResponse.model_validate(
            VALID_QUESTION_ONLY_RESPONSE
        )

        res = client.post(
            "/courses/generate",
            json={"question": "Explique la régression", "mode": "question_only"},
        )

        assert res.status_code == 200
        assert mock_generate.call_args.kwargs["mode"] == "question_only"


def test_generate_course_file_question_explicit_mode(client):
    """Mode file_question forcé explicitement avec filename."""
    with patch(
        "app.api.routes.generate_course_from_question", new_callable=AsyncMock
    ) as mock_generate:
        from app.api.schemas import CourseGenerationResponse

        mock_generate.return_value = CourseGenerationResponse.model_validate(VALID_RESPONSE)

        res = client.post(
            "/courses/generate",
            json={"question": "question", "filename": "doc.pdf", "mode": "file_question"},
        )

        assert res.status_code == 200
        assert mock_generate.call_args.kwargs["mode"] == "file_question"
        assert mock_generate.call_args.kwargs["filename"] == "doc.pdf"


def test_generate_course_mode_overrides_filename_detection(client):
    """mode explicite prime sur l'auto-détection par filename."""
    with patch(
        "app.api.routes.generate_course_from_question", new_callable=AsyncMock
    ) as mock_generate:
        from app.api.schemas import CourseGenerationResponse

        mock_generate.return_value = CourseGenerationResponse.model_validate(
            VALID_QUESTION_ONLY_RESPONSE
        )

        res = client.post(
            "/courses/generate",
            json={"question": "question", "filename": "doc.pdf", "mode": "question_only"},
        )

        assert res.status_code == 200
        assert mock_generate.call_args.kwargs["mode"] == "question_only"

# ─── Génération en deux temps : /courses/plan → /courses/generate/from-plan ───

import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from app.api.schemas import ApiPlannedSection
from app.schemas.course_generation import CoursePlanSchema


class _FakeSessionFactory:
    """Remplace app.state.db_session_factory : `async with factory() as db` sans PostgreSQL."""

    def __call__(self):
        return self

    async def __aenter__(self):
        return AsyncMock()

    async def __aexit__(self, *exc):
        return False


def _planned_json(order: int, type_: str = "development", title: str | None = None) -> dict:
    return {
        "type": type_, "title": title or f"Section {order}",
        "objective": "obj", "subtopics": ["a", "b"], "order": order,
    }


def _plan_row(**overrides) -> SimpleNamespace:
    row = dict(
        id=uuid.uuid4(), question="Q", mode="file_question", filenames=["doc.pdf"],
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    row.update(overrides)
    return SimpleNamespace(**row)


@pytest.fixture
def plan_client(client):
    app.state.db_session_factory = _FakeSessionFactory()
    return client


def test_create_course_plan_success(plan_client):
    plan = CoursePlanSchema.model_validate({
        "meta": {"title": "T", "subject": "S", "language": "fr"},
        "planned_sections": [_planned_json(1, "introduction"), _planned_json(2)],
        "coverage_notes": "RAS",
    })
    saved = _plan_row()
    with patch("app.api.routes.generate_course_plan", new_callable=AsyncMock) as mock_plan, patch(
        "app.api.routes.course_plan_repository.save", new_callable=AsyncMock
    ) as mock_save:
        mock_plan.return_value = (plan, {"chunks": []})
        mock_save.return_value = saved

        res = plan_client.post("/courses/plan", json={"question": "Q", "filename": "doc.pdf"})

    assert res.status_code == 200
    body = res.json()
    assert body["plan_id"] == str(saved.id)
    assert body["mode"] == "file_question"
    assert [s["title"] for s in body["sections"]] == ["Section 1", "Section 2"]
    assert mock_save.call_args.kwargs["retrieval_context"] == {"chunks": []}
    assert mock_save.call_args.kwargs["filenames"] == ["doc.pdf"]


def test_create_course_plan_question_too_long_rejected(plan_client):
    res = plan_client.post("/courses/plan", json={"question": "a" * 3000})
    assert res.status_code == 413


def test_create_course_plan_gemini_invalid_response(plan_client):
    with patch("app.api.routes.generate_course_plan", new_callable=AsyncMock) as mock_plan:
        mock_plan.side_effect = GeminiInvalidResponseError("plan invalide")
        res = plan_client.post("/courses/plan", json={"question": "Q"})
    assert res.status_code == 502


def _from_plan_payload(plan_id, sections) -> dict:
    return {"plan_id": str(plan_id), "sections": sections}


def test_generate_from_plan_success_uses_edited_sections(plan_client):
    from app.api.schemas import CourseGenerationResponse

    row = _plan_row()
    edited = [_planned_json(1, "introduction", "Intro"), _planned_json(2, title="Titre modifié")]
    with patch(
        "app.api.routes.course_plan_repository.get_by_id", new_callable=AsyncMock, return_value=row
    ), patch(
        "app.api.routes.course_plan_repository.mark_generated", new_callable=AsyncMock
    ) as mock_mark, patch(
        "app.api.routes.course_session_repository.save", new_callable=AsyncMock
    ) as mock_session_save, patch(
        "app.api.routes.generate_course_from_validated_plan", new_callable=AsyncMock
    ) as mock_generate:
        mock_generate.return_value = CourseGenerationResponse.model_validate(VALID_RESPONSE)

        res = plan_client.post("/courses/generate/from-plan", json=_from_plan_payload(row.id, edited))

    assert res.status_code == 200
    assert res.json()["mode"] == "file_question"
    sent = mock_generate.call_args.kwargs["edited_sections"]
    assert [s.title for s in sent] == ["Intro", "Titre modifié"]
    assert mock_generate.call_args.kwargs["plan_row"] is row
    mock_session_save.assert_awaited_once()
    mock_mark.assert_awaited_once()


def test_generate_from_plan_unknown_plan_404(plan_client):
    with patch("app.api.routes.course_plan_repository.get_by_id", new_callable=AsyncMock, return_value=None):
        res = plan_client.post(
            "/courses/generate/from-plan", json=_from_plan_payload(uuid.uuid4(), [_planned_json(1)])
        )
    assert res.status_code == 404


def test_generate_from_plan_expired_plan_410(plan_client):
    row = _plan_row(expires_at=datetime.now(timezone.utc) - timedelta(minutes=1))
    with patch("app.api.routes.course_plan_repository.get_by_id", new_callable=AsyncMock, return_value=row):
        res = plan_client.post("/courses/generate/from-plan", json=_from_plan_payload(row.id, [_planned_json(1)]))
    assert res.status_code == 410


def test_generate_from_plan_too_many_sections_422(plan_client):
    sections = [_planned_json(i) for i in range(1, 82)]
    res = plan_client.post("/courses/generate/from-plan", json=_from_plan_payload(uuid.uuid4(), sections))
    assert res.status_code == 422


@pytest.mark.parametrize("sections", [
    [],
    [_planned_json(1, "introduction")],  # aucune section development
    [{**_planned_json(1), "title": "   "}],
    [{**_planned_json(1), "type": "inconnu"}],
    [{**_planned_json(1), "order": 0}],
])
def test_generate_from_plan_invalid_sections_422(plan_client, sections):
    res = plan_client.post("/courses/generate/from-plan", json=_from_plan_payload(uuid.uuid4(), sections))
    assert res.status_code == 422


def test_generate_from_plan_invalid_uuid_422(plan_client):
    res = plan_client.post("/courses/generate/from-plan", json={"plan_id": "pas-un-uuid", "sections": [_planned_json(1)]})
    assert res.status_code == 422


# ─── Assistance IA sur le plan : /courses/plan/refine-section et /courses/plan/more-sections ───


def _refine_payload(plan_id, **overrides) -> dict:
    payload = {
        "plan_id": str(plan_id),
        "section": _planned_json(2, title="Principe"),
        "sections": [_planned_json(1, "introduction", "Intro"), _planned_json(2, title="Principe")],
    }
    payload.update(overrides)
    return payload


def test_refine_section_success_passes_instructions(plan_client):
    row = _plan_row()
    refined = ApiPlannedSection(**_planned_json(2, title="Principe"))
    with patch(
        "app.api.routes.course_plan_repository.get_by_id", new_callable=AsyncMock, return_value=row
    ), patch("app.api.routes.refine_planned_section", new_callable=AsyncMock, return_value=refined) as mock_refine:
        res = plan_client.post(
            "/courses/plan/refine-section", json=_refine_payload(row.id, instructions="  ajoute X  ")
        )

    assert res.status_code == 200
    assert res.json()["title"] == "Principe"
    assert mock_refine.call_args.kwargs["instructions"] == "ajoute X"
    assert mock_refine.call_args.kwargs["plan_row"] is row
    assert len(mock_refine.call_args.kwargs["outline"]) == 2


def test_refine_section_blank_instructions_become_none(plan_client):
    row = _plan_row()
    refined = ApiPlannedSection(**_planned_json(2))
    with patch(
        "app.api.routes.course_plan_repository.get_by_id", new_callable=AsyncMock, return_value=row
    ), patch("app.api.routes.refine_planned_section", new_callable=AsyncMock, return_value=refined) as mock_refine:
        res = plan_client.post("/courses/plan/refine-section", json=_refine_payload(row.id, instructions="   "))

    assert res.status_code == 200
    assert mock_refine.call_args.kwargs["instructions"] is None


def test_refine_section_unknown_plan_404(plan_client):
    with patch("app.api.routes.course_plan_repository.get_by_id", new_callable=AsyncMock, return_value=None):
        res = plan_client.post("/courses/plan/refine-section", json=_refine_payload(uuid.uuid4()))
    assert res.status_code == 404


def test_refine_section_expired_plan_410(plan_client):
    row = _plan_row(expires_at=datetime.now(timezone.utc) - timedelta(minutes=1))
    with patch("app.api.routes.course_plan_repository.get_by_id", new_callable=AsyncMock, return_value=row):
        res = plan_client.post("/courses/plan/refine-section", json=_refine_payload(row.id))
    assert res.status_code == 410


def test_refine_section_instructions_too_long_422(plan_client):
    res = plan_client.post(
        "/courses/plan/refine-section", json=_refine_payload(uuid.uuid4(), instructions="a" * 1001)
    )
    assert res.status_code == 422


def test_refine_section_gemini_quota_429(plan_client):
    row = _plan_row()
    with patch(
        "app.api.routes.course_plan_repository.get_by_id", new_callable=AsyncMock, return_value=row
    ), patch(
        "app.api.routes.refine_planned_section", new_callable=AsyncMock, side_effect=GeminiQuotaExceededError("quota")
    ):
        res = plan_client.post("/courses/plan/refine-section", json=_refine_payload(row.id))
    assert res.status_code == 429


def test_more_sections_success(plan_client):
    row = _plan_row()
    created = MoreSectionsResult(sections=[ApiPlannedSection(**_planned_json(3, title="Nouveau"))])
    with patch(
        "app.api.routes.course_plan_repository.get_by_id", new_callable=AsyncMock, return_value=row
    ), patch("app.api.routes.generate_more_sections", new_callable=AsyncMock, return_value=created) as mock_more:
        res = plan_client.post(
            "/courses/plan/more-sections",
            json={"plan_id": str(row.id), "sections": [_planned_json(1), _planned_json(2, "next_steps", "Suite")]},
        )

    assert res.status_code == 200
    assert [s["title"] for s in res.json()["sections"]] == ["Nouveau"]
    assert len(mock_more.call_args.kwargs["current_sections"]) == 2


def test_more_sections_gemini_invalid_response_502(plan_client):
    row = _plan_row()
    with patch(
        "app.api.routes.course_plan_repository.get_by_id", new_callable=AsyncMock, return_value=row
    ), patch(
        "app.api.routes.generate_more_sections", new_callable=AsyncMock, side_effect=GeminiInvalidResponseError("x")
    ):
        res = plan_client.post(
            "/courses/plan/more-sections", json={"plan_id": str(row.id), "sections": [_planned_json(1)]}
        )
    assert res.status_code == 502


def test_more_sections_unknown_plan_404(plan_client):
    with patch("app.api.routes.course_plan_repository.get_by_id", new_callable=AsyncMock, return_value=None):
        res = plan_client.post(
            "/courses/plan/more-sections", json={"plan_id": str(uuid.uuid4()), "sections": [_planned_json(1)]}
        )
    assert res.status_code == 404


def test_more_sections_empty_sections_422(plan_client):
    res = plan_client.post("/courses/plan/more-sections", json={"plan_id": str(uuid.uuid4()), "sections": []})
    assert res.status_code == 422


# ─── GET /courses/plans (plans en cours) et GET /courses/plans/{plan_id} ───


def _stored_plan(**overrides) -> SimpleNamespace:
    plan = {
        "meta": {"title": "Transformateurs", "subject": "Électrotechnique", "language": "fr"},
        "planned_sections": [_planned_json(1, "introduction", "Intro"), _planned_json(2, title="Principe")],
        "coverage_notes": "RAS",
    }
    row = dict(
        id=uuid.uuid4(), question="Explique les transformateurs", mode="file_question", filenames=["doc.pdf"],
        plan=plan, created_at=datetime.now(timezone.utc),
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    row.update(overrides)
    return SimpleNamespace(**row)


def test_list_pending_plans_maps_rows_and_pagination(plan_client):
    row = _stored_plan()
    with patch(
        "app.api.routes.course_plan_repository.list_pending", new_callable=AsyncMock, return_value=([row], 25)
    ) as mock_list:
        res = plan_client.get("/courses/plans?page=2&limit=10")

    assert res.status_code == 200
    body = res.json()
    assert body["data"][0]["plan_id"] == str(row.id)
    assert body["data"][0]["title"] == "Transformateurs"
    assert body["data"][0]["sections_count"] == 2
    assert body["meta"] == {"page": 2, "limit": 10, "total": 25, "totalPages": 3}
    assert mock_list.call_args.kwargs["page"] == 2
    assert mock_list.call_args.kwargs["limit"] == 10


def test_list_pending_plans_empty(plan_client):
    with patch("app.api.routes.course_plan_repository.list_pending", new_callable=AsyncMock, return_value=([], 0)):
        res = plan_client.get("/courses/plans")

    assert res.status_code == 200
    assert res.json()["data"] == []
    assert res.json()["meta"]["total"] == 0


def test_get_plan_detail_returns_sections_and_original_request(plan_client):
    row = _stored_plan()
    with patch("app.api.routes.course_plan_repository.get_by_id", new_callable=AsyncMock, return_value=row):
        res = plan_client.get(f"/courses/plans/{row.id}")

    assert res.status_code == 200
    body = res.json()
    assert body["plan_id"] == str(row.id)
    assert [s["title"] for s in body["sections"]] == ["Intro", "Principe"]
    assert body["question"] == "Explique les transformateurs"
    assert body["filenames"] == ["doc.pdf"]
    assert body["mode"] == "file_question"


def test_get_plan_detail_unknown_404(plan_client):
    with patch("app.api.routes.course_plan_repository.get_by_id", new_callable=AsyncMock, return_value=None):
        assert plan_client.get(f"/courses/plans/{uuid.uuid4()}").status_code == 404


def test_get_plan_detail_expired_410(plan_client):
    row = _stored_plan(expires_at=datetime.now(timezone.utc) - timedelta(minutes=1))
    with patch("app.api.routes.course_plan_repository.get_by_id", new_callable=AsyncMock, return_value=row):
        assert plan_client.get(f"/courses/plans/{row.id}").status_code == 410


def test_get_plan_detail_invalid_uuid_422(plan_client):
    assert plan_client.get("/courses/plans/pas-un-uuid").status_code == 422
