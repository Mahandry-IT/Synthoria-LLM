"""Câblage de `video_search_queries` : chaque site d'appel doit le transmettre à
`attach_verified_videos`, et `video_suggestions` ne doit plus exister nulle part."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.schemas.course_generation import CourseGenerationSchema
from app.services import course_generator, course_plan_generator
from app.services.course_generator import generate_course_from_question
from app.services.course_plan_generator import generate_course_from_validated_plan
from app.schemas.course_generation import SectionsBatchSchema
from tests.test_course_generator import VALID_STRUCTURED_ANSWER_FILE
from tests.test_course_plan_generator import _api_sections, _dev_section, _plan_row, _wrap_up_structured


def test_video_suggestions_no_longer_exists_on_the_gemini_schema():
    fields = CourseGenerationSchema.model_fields
    assert "video_search_queries" in fields
    assert "video_suggestions" not in fields


@pytest.mark.asyncio
async def test_direct_mode_call_site_forwards_queries_and_session_factory(monkeypatch):
    structured = {**VALID_STRUCTURED_ANSWER_FILE, "video_search_queries": ["ampli op cours"]}
    gemini = AsyncMock()
    gemini.format_structured.return_value = structured
    settings = Settings(gemini_api_key="k", gemini_use_search_grounding=False)
    vector_store = AsyncMock()
    vector_store.search.return_value = []
    attach = AsyncMock(side_effect=lambda response, *_a, **_k: response)
    monkeypatch.setattr(course_generator, "attach_verified_videos", attach)
    sentinel_factory = object()

    await generate_course_from_question(
        question="Q ?", vector_store=vector_store, gemini_client=gemini, settings=settings,
        mode="file_question", db_session_factory=sentinel_factory,
    )

    attach.assert_awaited_once()
    assert attach.await_args.kwargs["search_queries"] == ["ampli op cours"]
    assert attach.await_args.kwargs["db_session_factory"] is sentinel_factory


@pytest.mark.asyncio
async def test_grounded_mode_call_site_forwards_queries(monkeypatch):
    structured = {**VALID_STRUCTURED_ANSWER_FILE, "video_search_queries": ["méthode ampli op"]}
    gemini = AsyncMock()
    gemini.search_grounded.return_value = ("réponse", [])
    gemini.format_structured.return_value = structured
    settings = Settings(gemini_api_key="k", gemini_use_search_grounding=True)
    vector_store = AsyncMock()
    vector_store.search.return_value = []
    attach = AsyncMock(side_effect=lambda response, *_a, **_k: response)
    monkeypatch.setattr(course_generator, "attach_verified_videos", attach)

    await generate_course_from_question(
        question="Q ?", vector_store=vector_store, gemini_client=gemini, settings=settings, mode="file_question",
    )

    attach.assert_awaited_once()
    assert attach.await_args.kwargs["search_queries"] == ["méthode ampli op"]
    assert attach.await_args.kwargs["db_session_factory"] is None


@pytest.mark.asyncio
async def test_from_plan_call_site_forwards_wrap_up_queries(monkeypatch):
    wrap_up = {**_wrap_up_structured(), "video_search_queries": ["exercices corrigés ampli op"]}

    async def format_structured(raw_answer, system_instruction, *, response_schema=None):
        if response_schema is SectionsBatchSchema:
            return {"sections": [_dev_section("A")]}
        return wrap_up

    client = AsyncMock()
    client.format_structured.side_effect = format_structured
    attach = AsyncMock(side_effect=lambda response, *_a, **_k: response)
    monkeypatch.setattr(course_plan_generator, "attach_verified_videos", attach)
    settings = Settings(gemini_api_key="k")
    sentinel_factory = object()

    await generate_course_from_validated_plan(
        _plan_row(), _api_sections(["A"]), client, settings, db_session_factory=sentinel_factory,
    )

    attach.assert_awaited_once()
    assert attach.await_args.kwargs["search_queries"] == ["exercices corrigés ampli op"]
    assert attach.await_args.kwargs["db_session_factory"] is sentinel_factory
