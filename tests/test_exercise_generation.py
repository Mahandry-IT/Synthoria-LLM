from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.core.exceptions import GeminiInvalidResponseError
from app.schemas.exercise_generation import ExerciseBatch, exercise_batch_json_schema
from app.services import exercise_generator as gen
from tests.exercise_fixtures import TEXT_BLOCK, course, draft

SETTINGS = Settings(gemini_api_key="k")
IDS = {"s1", "s2"}


def test_json_schema_requires_every_exercise_field_and_three_hints():
    schema = exercise_batch_json_schema(3)
    item = schema["$defs"]["ExerciseDraft"]
    assert set(item["required"]) == set(item["properties"])
    assert {"section_id", "statement", "hints", "expected_solution"} <= set(item["required"])
    assert item["properties"]["hints"]["minItems"] == 3 and item["properties"]["hints"]["maxItems"] == 3
    assert item["properties"]["statement"]["minItems"] == 1
    assert schema["properties"]["exercises"]["maxItems"] == 3


@pytest.mark.parametrize("depth,cap", [("express", 2), ("standard", 3), ("approfondi", 4)])
def test_cap_per_mode(depth, cap):
    assert gen.max_exercises(depth, SETTINGS) == cap
    rows = gen.validate_batch({"exercises": [draft(title=f"T{i}") for i in range(8)]}, valid_section_ids=IDS, cap=cap)
    assert len(rows) == cap


def test_unknown_section_id_is_discarded_but_null_is_allowed():
    rows = gen.validate_batch(
        {"exercises": [draft(section_id="inconnue"), draft(section_id=None, title="Transverse")]},
        valid_section_ids=IDS, cap=4,
    )
    assert [r.title for r in rows] == ["Transverse"]


def test_empty_statement_blocks_are_dropped_and_empty_statement_discards_exercise():
    empty_table = {"type": "table"}
    kept = gen.validate_batch(
        {"exercises": [draft(statement=[empty_table, TEXT_BLOCK]), draft(title="Vide", statement=[empty_table])]},
        valid_section_ids=IDS, cap=4,
    )
    assert len(kept) == 1 and [b["type"] for b in kept[0].statement] == ["text"]


def test_exercise_needs_exactly_three_hints():
    rows = gen.validate_batch(
        {"exercises": [draft(hints=["a", "b"]), draft(title="Ok", hints=["a", "b", "c", "d"])]},
        valid_section_ids=IDS, cap=4,
    )
    assert [r.title for r in rows] == ["Ok"] and rows[0].hints == ["a", "b", "c"]


def test_all_invalid_is_an_invalid_response_but_empty_list_is_legitimate():
    with pytest.raises(GeminiInvalidResponseError):
        gen.validate_batch({"exercises": [draft(section_id="x")]}, valid_section_ids=IDS, cap=4)
    assert gen.validate_batch({"exercises": []}, valid_section_ids=IDS, cap=4) == []
    with pytest.raises(GeminiInvalidResponseError):
        gen.validate_batch({"exercises": [{"title": "incomplet"}]}, valid_section_ids=IDS, cap=4)


def test_image_blocks_are_not_kept():
    rows = gen.validate_batch(
        {"exercises": [draft(statement=[{"type": "image", "image_caption": "x", "image_source": "web"}, TEXT_BLOCK])]},
        valid_section_ids=IDS, cap=4,
    )
    assert [b["type"] for b in rows[0].statement] == ["text"]


def test_worked_example_blocks_are_not_kept_so_the_solution_cannot_leak():
    leaky = {"type": "worked_example", "worked_example": {"statement": "q", "steps": ["a"], "result": "42"}}
    rows = gen.validate_batch(
        {"exercises": [draft(statement=[leaky, TEXT_BLOCK])]}, valid_section_ids=IDS, cap=4,
    )
    assert [b["type"] for b in rows[0].statement] == ["text"]
    # Un énoncé réduit à un exemple résolu n'est plus un énoncé : l'exercice est écarté.
    with pytest.raises(GeminiInvalidResponseError):
        gen.validate_batch({"exercises": [draft(statement=[leaky])]}, valid_section_ids=IDS, cap=4)


def test_generation_prompt_forbids_worked_example():
    assert "WORKED_EXAMPLE" in gen._SYSTEM_INSTRUCTION


@pytest.mark.asyncio
async def test_whole_batch_is_one_gemini_call():
    client = AsyncMock()
    client.format_structured.return_value = {"exercises": [draft(), draft(section_id="s2", title="Deux")]}

    rows = await gen.generate_exercises(course("standard"), client, SETTINGS)

    assert len(rows) == 2
    client.format_structured.assert_awaited_once()
    kwargs = client.format_structured.await_args.kwargs
    assert "id=s1" in kwargs["raw_answer"] and "id=s2" in kwargs["raw_answer"]
    assert kwargs["response_schema"]["properties"]["exercises"]["maxItems"] == 3


@pytest.mark.asyncio
async def test_mode_cap_is_sent_and_enforced():
    client = AsyncMock()
    client.format_structured.return_value = {"exercises": [draft(title=f"T{i}") for i in range(5)]}
    rows = await gen.generate_exercises(course("express"), client, SETTINGS)
    assert len(rows) == 2
    assert client.format_structured.await_args.kwargs["response_schema"]["properties"]["exercises"]["maxItems"] == 2


@pytest.mark.asyncio
async def test_course_without_usable_section_makes_no_call():
    client = AsyncMock()
    assert await gen.generate_exercises(course(sections=[]), client, SETTINGS) == []
    client.format_structured.assert_not_awaited()


def test_course_is_ready_rejects_missing_or_incomplete_sections():
    assert gen.course_is_ready(course())
    assert not gen.course_is_ready(course(sections=[]))
    assert not gen.course_is_ready(course(sections=[{"id": "s1", "title": "Vide", "subsections": []}]))


def test_batch_model_accepts_valid_payload():
    assert ExerciseBatch.model_validate({"exercises": [draft()]}).exercises[0].title
