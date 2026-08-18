from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.core.exceptions import GeminiInvalidResponseError
from app.services.course_generator import generate_course_from_question

VALID_STRUCTURED_ANSWER = {
    "mode": "file_question",
    "format": "focused_answer",
    "meta": {
        "title": "Titre",
        "subject": "Sujet",
        "language": "fr",
        "generated_at": "2026-08-18T10:00:00Z",
    },
    "sources": [{"type": "file", "label": "doc.pdf", "reference": "page 2"}],
    "answer": {
        "quoi": "définition",
        "pourquoi": "raison",
        "comment": "mécanisme",
        "worked_example": {"statement": "énoncé", "steps": ["étape 1"], "result": "résultat"},
        "key_points": ["point 1"],
    },
    "summary": "résumé",
    "next_steps": ["étape suivante"],
}


@pytest.fixture
def settings() -> Settings:
    return Settings(gemini_api_key="fake-key")


@pytest.fixture
def vector_store():
    store = AsyncMock()
    store.search.return_value = [
        {"content": "extrait pertinent", "metadata": {"filename": "doc.pdf", "page": 2}, "distance": 0.1}
    ]
    return store


@pytest.fixture
def gemini_client():
    client = AsyncMock()
    client.search_grounded.return_value = ("réponse brute", [{"type": "web", "label": "W", "reference": "https://w"}])
    client.format_structured.return_value = VALID_STRUCTURED_ANSWER
    return client


@pytest.mark.asyncio
async def test_generate_course_file_question_happy_path(settings, vector_store, gemini_client):
    result = await generate_course_from_question(
        question="Comment fonctionne un transformateur ?",
        vector_store=vector_store,
        gemini_client=gemini_client,
        settings=settings,
        mode="file_question",
    )

    assert result.mode == "file_question"
    assert result.format == "focused_answer"
    assert result.answer is not None
    vector_store.search.assert_awaited_once()
    gemini_client.search_grounded.assert_awaited_once()
    gemini_client.format_structured.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_course_filters_by_filename(settings, gemini_client):
    vector_store = AsyncMock()
    vector_store.search.return_value = [
        {"content": "a", "metadata": {"filename": "doc1.pdf", "page": 1}, "distance": 0.1},
        {"content": "b", "metadata": {"filename": "doc2.pdf", "page": 1}, "distance": 0.2},
    ]

    await generate_course_from_question(
        question="question",
        vector_store=vector_store,
        gemini_client=gemini_client,
        settings=settings,
        filename="doc1.pdf",
    )

    prompt_used = gemini_client.search_grounded.call_args.kwargs["prompt"]
    assert "doc1.pdf" in prompt_used
    assert "doc2.pdf" not in prompt_used


@pytest.mark.asyncio
async def test_generate_course_question_only_skips_retrieval(settings, gemini_client):
    vector_store = AsyncMock()

    await generate_course_from_question(
        question="question",
        vector_store=vector_store,
        gemini_client=gemini_client,
        settings=settings,
        mode="question_only",
    )

    vector_store.search.assert_not_awaited()


@pytest.mark.asyncio
async def test_generate_course_invalid_structured_response_raises(settings, vector_store):
    gemini_client = AsyncMock()
    gemini_client.search_grounded.return_value = ("réponse brute", [])
    gemini_client.format_structured.return_value = {"mode": "file_question"}  # incomplet, invalide

    with pytest.raises(GeminiInvalidResponseError):
        await generate_course_from_question(
            question="question",
            vector_store=vector_store,
            gemini_client=gemini_client,
            settings=settings,
        )