from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.exceptions import (
    GeminiInvalidResponseError,
    GeminiQuotaExceededError,
    GeminiUnavailableError,
)
from app.main import app

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