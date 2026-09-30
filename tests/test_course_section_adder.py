from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.core.exceptions import GeminiInvalidResponseError, GeminiUnavailableError
from app.schemas.course_generation import SectionType
from app.services.course_section_adder import add_course_sections


def _row(
    *,
    mode: str = "file_question",
    filenames: list[str] | None = None,
    sections: list[dict] | None = None,
    next_steps: list[str] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        question="Comment fonctionne un transformateur ?",
        mode=mode,
        filenames=filenames or [],
        gemini_response={"sections": sections or [], "next_steps": next_steps or []},
    )


def _good_section(title: str = "Nouvelle notion") -> dict:
    return {
        "type": "development", "title": title, "blocks": [],
        "subsections": [
            {"title": "Pourquoi", "blocks": [{"type": "text", "text": "raison"}]},
            {"title": "Quoi", "blocks": [{"type": "text", "text": "définition"}]},
            {"title": "Comment", "blocks": [{"type": "text", "text": "mécanisme"}]},
        ],
    }


@pytest.mark.asyncio
async def test_user_instructions_drive_the_topic_and_leave_next_steps_untouched():
    gemini = AsyncMock()
    gemini.format_structured.return_value = {"sections": [_good_section()]}
    vector_store = AsyncMock()
    vector_store.search.return_value = []
    settings = Settings(gemini_api_key="k")
    row = _row(next_steps=["Approfondir le rendement"])

    sections, next_steps = await add_course_sections(row, "Parle des pertes fer", gemini, vector_store, settings)

    assert sections[0].title == "Nouvelle notion"
    assert sections[0].type is SectionType.DEVELOPMENT
    prompt = gemini.format_structured.await_args.kwargs["raw_answer"]
    assert "Parle des pertes fer" in prompt
    # Les next_steps existants ne viennent pas de ces instructions : inchangés.
    assert next_steps == ["Approfondir le rendement"]


@pytest.mark.asyncio
async def test_empty_instructions_use_next_steps_as_topic_and_clear_them():
    gemini = AsyncMock()
    gemini.format_structured.return_value = {"sections": [_good_section("Le rendement"), _good_section("Les pertes")]}
    vector_store = AsyncMock()
    settings = Settings(gemini_api_key="k")
    row = _row(next_steps=["Le rendement", "Les pertes fer"])

    sections, next_steps = await add_course_sections(row, "", gemini, vector_store, settings)

    assert [s.title for s in sections] == ["Le rendement", "Les pertes"]
    prompt = gemini.format_structured.await_args.kwargs["raw_answer"]
    assert "Le rendement" in prompt and "Les pertes fer" in prompt
    # Les pistes ont toutes été transformées en sections : plus rien à suggérer.
    assert next_steps == []


@pytest.mark.asyncio
async def test_empty_instructions_and_no_next_steps_let_the_model_propose_topics():
    gemini = AsyncMock()
    gemini.format_structured.return_value = {"sections": [_good_section()]}
    vector_store = AsyncMock()
    settings = Settings(gemini_api_key="k")
    row = _row(next_steps=[])

    sections, next_steps = await add_course_sections(row, "", gemini, vector_store, settings)

    assert len(sections) == 1
    prompt = gemini.format_structured.await_args.kwargs["raw_answer"]
    assert "détermine toi-même" in prompt
    assert next_steps == []


@pytest.mark.asyncio
async def test_prompt_lists_existing_section_titles_to_avoid_duplicates():
    gemini = AsyncMock()
    gemini.format_structured.return_value = {"sections": [_good_section()]}
    vector_store = AsyncMock()
    settings = Settings(gemini_api_key="k")
    row = _row(sections=[{"title": "Introduction"}, {"title": "Le modèle"}])

    await add_course_sections(row, "Sujet libre", gemini, vector_store, settings)

    prompt = gemini.format_structured.await_args.kwargs["raw_answer"]
    assert "Introduction" in prompt and "Le modèle" in prompt


@pytest.mark.asyncio
async def test_question_only_mode_uses_search_grounded_not_the_vector_store():
    gemini = AsyncMock()
    gemini.search_grounded.return_value = ("synthèse web", [{"type": "web", "reference": "https://x"}])
    gemini.format_structured.return_value = {"sections": [_good_section()]}
    vector_store = AsyncMock()
    settings = Settings(gemini_api_key="k")

    await add_course_sections(_row(mode="question_only"), "Sujet libre", gemini, vector_store, settings)

    gemini.search_grounded.assert_awaited_once()
    vector_store.search.assert_not_called()


@pytest.mark.asyncio
async def test_no_sections_returned_raises_invalid_response():
    gemini = AsyncMock()
    gemini.format_structured.return_value = {"sections": []}
    vector_store = AsyncMock()
    vector_store.search.return_value = []
    settings = Settings(gemini_api_key="k")

    with pytest.raises(GeminiInvalidResponseError):
        await add_course_sections(_row(), "Sujet libre", gemini, vector_store, settings)


@pytest.mark.asyncio
async def test_gemini_service_error_propagates_never_silently_falls_back():
    gemini = AsyncMock()
    gemini.format_structured.side_effect = GeminiUnavailableError("indisponible")
    vector_store = AsyncMock()
    vector_store.search.return_value = []
    settings = Settings(gemini_api_key="k")

    with pytest.raises(GeminiUnavailableError):
        await add_course_sections(_row(), "Sujet libre", gemini, vector_store, settings)
