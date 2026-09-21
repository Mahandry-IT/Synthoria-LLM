import copy
from unittest.mock import AsyncMock

import pytest

from app.api.schemas import CourseAnswer, CourseGenerationResponse
from app.services.course_generator import (
    _validate_and_map,
    _validate_and_map_with_retry,
    answer_intro_overlap,
    ensure_distinct_direct_answer,
)
from app.schemas.course_generation import CourseGenerationSchema

INTRO = "Un transformateur est une machine électrique statique qui transfère de l'énergie entre deux circuits par induction."
ANSWER = "Le transformateur change la tension alternative grâce au rapport entre ses nombres de spires."


def _section(type_: str, title: str, text: str) -> dict:
    return {
        "type": type_, "title": title, "blocks": [],
        "subsections": [{"title": "Quoi", "blocks": [{"type": "text", "text": text}]}],
    }


def _structured(direct_summary: str = ANSWER) -> dict:
    return {
        "mode": "question_only",
        "format": "focused_answer",
        "meta": {"title": "T", "subject": "S", "language": "fr", "generated_at": "2026-08-19T10:00:00Z"},
        "sources": [],
        "sections": [_section("introduction", "Introduction", INTRO), _section("development", "Le transformateur", "def")],
        "direct_answer": {"summary": direct_summary, "key_points": ["Rapport de spires"], "blocks": []},
        "quiz": [],
        "confidence": "high",
        "unconfirmed_points": [],
    }


def test_answer_comes_from_direct_answer_not_first_section():
    result = _validate_and_map(_structured(), "question_only")

    assert result.answer.summary == ANSWER
    assert result.answer.key_points == ["Rapport de spires"]
    assert result.answer.quoi is None
    assert result.answer.summary != result.sections[0].quoi


def test_answer_falls_back_without_copying_first_section():
    data = _structured()
    del data["direct_answer"]

    result = _validate_and_map(data, "question_only")

    assert result.answer is not None
    assert INTRO not in result.answer.summary and "def" not in result.answer.summary


def test_legacy_answer_format_still_valid():
    legacy = {
        "quoi": "q", "pourquoi": "p", "comment": "c",
        "worked_example": {"statement": "s", "steps": [], "result": "r"},
        "key_points": [],
    }
    answer = CourseAnswer.model_validate(legacy)
    assert answer.quoi == "q" and answer.summary == "" and answer.blocks == []


def test_overlap_detects_copied_intro():
    copied = CourseGenerationSchema.model_validate(_structured(INTRO))
    distinct = CourseGenerationSchema.model_validate(_structured())

    assert answer_intro_overlap(copied) > 0.5
    assert answer_intro_overlap(distinct) < 0.2


@pytest.mark.asyncio
async def test_guard_regenerates_only_direct_answer_when_duplicated():
    client = AsyncMock()
    client.format_structured.return_value = {"summary": ANSWER, "key_points": ["a"], "blocks": []}

    fixed = await ensure_distinct_direct_answer(
        _structured(INTRO), mode="question_only", gemini_client=client,
        system_instruction="i", context_prompt="ctx", max_overlap=0.5,
    )

    assert fixed["direct_answer"]["summary"] == ANSWER
    assert client.format_structured.await_count == 1
    assert client.format_structured.await_args.kwargs["response_schema"].__name__ == "DirectAnswer"


@pytest.mark.asyncio
async def test_guard_noop_when_distinct_or_on_failure():
    client = AsyncMock()
    data = _structured()
    assert await ensure_distinct_direct_answer(
        data, mode="question_only", gemini_client=client, system_instruction="i", context_prompt="c", max_overlap=0.5
    ) is data
    client.format_structured.assert_not_awaited()

    client.format_structured.side_effect = RuntimeError("boom")
    dup = _structured(INTRO)
    assert await ensure_distinct_direct_answer(
        copy.deepcopy(dup), mode="question_only", gemini_client=client, system_instruction="i",
        context_prompt="c", max_overlap=0.5,
    ) == dup


@pytest.mark.asyncio
async def test_retry_wrapper_applies_guard():
    client = AsyncMock()
    client.format_structured.return_value = {"summary": ANSWER, "key_points": [], "blocks": []}

    result = await _validate_and_map_with_retry(_structured(INTRO), "question_only", client, "i", "ctx")

    assert isinstance(result, CourseGenerationResponse)
    assert result.answer.summary == ANSWER
