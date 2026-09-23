from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.core.exceptions import GeminiInvalidResponseError, GeminiUnavailableError
from app.schemas.course_generation import SectionType
from app.services.section_regenerator import regenerate_section


def _row(mode: str = "file_question", filenames: list[str] | None = None) -> SimpleNamespace:
    return SimpleNamespace(question="Comment fonctionne un transformateur ?", mode=mode, filenames=filenames or [])


def _good_section(title: str = "Le transformateur") -> dict:
    return {
        "type": "development", "title": title, "blocks": [],
        "subsections": [
            {"title": "Pourquoi", "blocks": [{"type": "text", "text": "raison"}]},
            {"title": "Quoi", "blocks": [{"type": "text", "text": "définition"}]},
            {"title": "Comment", "blocks": [{"type": "text", "text": "mécanisme"}]},
        ],
    }


@pytest.mark.asyncio
async def test_file_question_mode_retrieves_chunks_and_returns_the_section():
    gemini = AsyncMock()
    gemini.reformulate_query.return_value = "transformateur"
    gemini.format_structured.return_value = {"sections": [_good_section("Le transformateur")]}
    vector_store = AsyncMock()
    vector_store.search.return_value = [{"content": "extrait", "metadata": {"filename": "doc.pdf", "page": 1}}]
    settings = Settings(gemini_api_key="k")

    result = await regenerate_section(_row(), "Le transformateur", gemini, vector_store, settings)

    assert result.title == "Le transformateur"
    assert result.type is SectionType.DEVELOPMENT
    vector_store.search.assert_awaited_once()
    gemini.search_grounded.assert_not_called()


@pytest.mark.asyncio
async def test_question_only_mode_uses_search_grounded_not_the_vector_store():
    gemini = AsyncMock()
    gemini.search_grounded.return_value = ("synthèse web", [{"type": "web", "reference": "https://x"}])
    gemini.format_structured.return_value = {"sections": [_good_section("Le transformateur")]}
    vector_store = AsyncMock()
    settings = Settings(gemini_api_key="k")

    result = await regenerate_section(_row(mode="question_only"), "Le transformateur", gemini, vector_store, settings)

    assert result.title == "Le transformateur"
    gemini.search_grounded.assert_awaited_once()
    vector_store.search.assert_not_called()


@pytest.mark.asyncio
async def test_returned_title_is_forced_to_the_requested_one():
    """Le titre demandé fait foi (id/position du client dépendent de ce titre exact)."""
    gemini = AsyncMock()
    gemini.reformulate_query.return_value = "q"
    gemini.format_structured.return_value = {"sections": [_good_section("Un autre titre halluciné")]}
    vector_store = AsyncMock()
    vector_store.search.return_value = []
    settings = Settings(gemini_api_key="k")

    result = await regenerate_section(_row(), "Le transformateur", gemini, vector_store, settings)

    assert result.title == "Le transformateur"


@pytest.mark.asyncio
async def test_no_sections_returned_raises_invalid_response():
    gemini = AsyncMock()
    gemini.reformulate_query.return_value = "q"
    gemini.format_structured.return_value = {"sections": []}
    vector_store = AsyncMock()
    vector_store.search.return_value = []
    settings = Settings(gemini_api_key="k")

    with pytest.raises(GeminiInvalidResponseError):
        await regenerate_section(_row(), "Le transformateur", gemini, vector_store, settings)


@pytest.mark.asyncio
async def test_invalid_structured_output_raises_invalid_response():
    gemini = AsyncMock()
    gemini.reformulate_query.return_value = "q"
    gemini.format_structured.return_value = {"not": "a valid section batch"}
    vector_store = AsyncMock()
    vector_store.search.return_value = []
    settings = Settings(gemini_api_key="k")

    with pytest.raises(GeminiInvalidResponseError):
        await regenerate_section(_row(), "Le transformateur", gemini, vector_store, settings)


@pytest.mark.asyncio
async def test_gemini_service_error_propagates_never_silently_falls_back():
    """Contrairement à la génération complète (best-effort), l'échec doit remonter tel quel."""
    gemini = AsyncMock()
    gemini.reformulate_query.return_value = "q"
    gemini.format_structured.side_effect = GeminiUnavailableError("indisponible")
    vector_store = AsyncMock()
    vector_store.search.return_value = []
    settings = Settings(gemini_api_key="k")

    with pytest.raises(GeminiUnavailableError):
        await regenerate_section(_row(), "Le transformateur", gemini, vector_store, settings)


@pytest.mark.asyncio
async def test_prompt_references_the_original_course_question_and_section_title():
    gemini = AsyncMock()
    gemini.reformulate_query.return_value = "q"
    gemini.format_structured.return_value = {"sections": [_good_section("Le transformateur")]}
    vector_store = AsyncMock()
    vector_store.search.return_value = []
    settings = Settings(gemini_api_key="k")

    await regenerate_section(_row(), "Le transformateur", gemini, vector_store, settings)

    prompt = gemini.format_structured.await_args.kwargs["raw_answer"]
    assert "Comment fonctionne un transformateur ?" in prompt and "Le transformateur" in prompt
