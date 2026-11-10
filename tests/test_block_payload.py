"""Un bloc sans la charge utile que son type annonce ne compte pas comme du contenu."""

import pytest

from app.schemas.course_generation import ContentBlock, Section
from app.services.course_generator import _map_block, is_incomplete_section_dict
from app.services.course_plan_generator import _has_content, _structural_gaps

_TABLE = {"caption": "c", "headers": ["a", "b"], "rows": [["1", "2"]]}


@pytest.mark.parametrize("raw", [
    {"type": "table"}, {"type": "table", "table": {"headers": [], "rows": []}},
    {"type": "list"}, {"type": "list", "list_items": ["  "]},
    {"type": "text"}, {"type": "text", "text": "   "}, {"type": "formula"}, {"type": "code"},
    {"type": "worked_example"}, {"type": "pitfall"}, {"type": "diagram"}, {"type": "chart"},
])
def test_block_without_payload_is_empty(raw):
    assert ContentBlock.model_validate(raw).has_payload() is False


@pytest.mark.parametrize("raw", [
    {"type": "table", "table": _TABLE}, {"type": "list", "list_items": ["a"]},
    {"type": "text", "text": "x"}, {"type": "definition", "text": "x"},
    {"type": "formula", "formula": {"latex": "x^2"}}, {"type": "code", "code": "print(1)"},
    {"type": "diagram", "diagram": {"kind": "flowchart", "mermaid": "flowchart TD; A-->B"}},
    {"type": "chart", "chart": {"kind": "bar", "labels": ["a"], "series": [{"name": "s", "values": [1]}]}},
    {"type": "image", "image_query": "x"},
])
def test_block_with_payload_is_not_empty(raw):
    assert ContentBlock.model_validate(raw).has_payload() is True


def test_map_block_drops_payloadless_block():
    assert _map_block(ContentBlock.model_validate({"type": "table"})) is None
    assert _map_block(ContentBlock.model_validate({"type": "table", "table": _TABLE})) is not None


def _section(quoi_blocks):
    return Section.model_validate({"type": "development", "title": "T", "subsections": [
        {"title": "Pourquoi", "blocks": [{"type": "text", "text": "p"}]},
        {"title": "Quoi", "blocks": quoi_blocks},
        {"title": "Comment", "blocks": [{"type": "text", "text": "c"}]},
    ]})


def test_subsection_with_only_payloadless_block_is_a_structural_gap():
    """Cas réel : « Quoi » = `{"type": "table"}` sans tableau, comptée à tort comme non vide."""
    assert _structural_gaps(_section([{"type": "table"}])) == ["Quoi"]
    assert _structural_gaps(_section([{"type": "table"}, {"type": "text", "text": "q"}])) == []


def test_section_with_only_payloadless_blocks_has_no_content():
    section = Section.model_validate({"type": "development", "title": "T", "blocks": [{"type": "list"}]})
    assert _has_content(section) is False


def _stored(quoi_blocks):
    return {"title": "T", "subsections": [
        {"title": "Pourquoi", "blocks": [{"type": "text", "text": "p"}]},
        {"title": "Quoi", "blocks": quoi_blocks},
        {"title": "Comment", "blocks": [{"type": "worked_example", "worked_example": {"statement": "s", "steps": ["a"], "result": "r"}}]},
    ]}


def test_stored_section_with_empty_quoi_is_incomplete():
    assert is_incomplete_section_dict(_stored([{"type": "table", "table": None}])) is True
    assert is_incomplete_section_dict(_stored([{"type": "list", "list_items": None}])) is True


def test_stored_complete_section_is_not_flagged():
    assert is_incomplete_section_dict(_stored([{"type": "table", "table": _TABLE}])) is False
