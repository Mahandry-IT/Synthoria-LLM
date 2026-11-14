"""Format « cycle » : schéma Pydantic et JSON Schema Gemini allégé, modèles d'API tolérants, anciens JSON intacts."""

import json
from typing import Any

import pytest

from app.api.schemas import CourseSection, WorkedExample
from app.schemas.course_generation import (
    COURSE_GENERATION_GEMINI_SCHEMA,
    CourseGenerationSchema,
    CoverageCompletionSchema,
    CycleSectionsBatchSchema,
    ExerciseCard,
    Section,
    SectionsBatchSchema,
)
from app.services.cycle_mapping import map_section_cycle
from tests.cycle_fixtures import cycle_section, old_format_section

OLD_FIELDS = ("subsections", "blocks", "challenge", "challenge_key_points", "faded_example", "check_questions", "pitfalls_to_avoid")


def _walk(node: Any):
    """Tous les sous-dictionnaires d'un schéma JSON."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item)


@pytest.fixture(scope="module")
def cycle_schema() -> dict:
    return CycleSectionsBatchSchema.model_json_schema()


# ─── JSON Schema envoyé à Gemini : format cycle ──────────────────────────────


def test_cycle_schema_requires_the_five_moments_of_a_development_section(cycle_schema):
    defs = cycle_schema["$defs"]

    assert set(defs["Section"]["required"]) >= {"type", "title", "cycle", "widget_brief", "recall_prompt", "key_takeaways"}
    assert set(defs["SectionCycle"]["required"]) == {"thread", "challenge", "steps", "vocabulary", "traps", "trial"}
    assert set(defs["CycleStep"]["required"]) >= {"text", "action", "support", "prediction"}
    assert "covered_subtopics" not in defs["Section"]["required"]  # n'a de sens que depuis un plan


def test_cycle_schema_cycle_is_never_null(cycle_schema):
    assert cycle_schema["$defs"]["Section"]["properties"]["cycle"]["$ref"].endswith("/SectionCycle")


def test_cycle_schema_is_slim_no_max_items_and_min_items_at_most_three(cycle_schema):
    nodes = list(_walk(cycle_schema))

    assert not any("maxItems" in node for node in nodes)
    assert all(node["minItems"] <= 3 for node in nodes if "minItems" in node)


def test_cycle_schema_has_no_min_items_on_sections_nor_steps(cycle_schema):
    """Un `minItems` sur ces gros tableaux d'objets a fait rejeter le schéma (HTTP 400) : le nombre est validé en Python."""
    assert "minItems" not in cycle_schema["properties"]["sections"]
    assert "minItems" not in cycle_schema["$defs"]["SectionCycle"]["properties"]["steps"]


def test_cycle_schema_exposes_only_cycle_fields_of_a_section(cycle_schema):
    properties = cycle_schema["$defs"]["Section"]["properties"]

    assert not set(OLD_FIELDS) & set(properties)
    assert {"cycle", "widget_brief", "recall_prompt", "key_takeaways", "covered_subtopics"} <= set(properties)
    assert not {"ContentBlock", "Subsection", "FadedExample", "PitfallToAvoid"} & set(cycle_schema["$defs"])


def test_cycle_schema_drops_titles_defaults_and_nullable_unions_of_optional_fields(cycle_schema):
    nodes = list(_walk(cycle_schema))
    support = cycle_schema["$defs"]["StepSupport"]["properties"]

    assert not any("default" in node for node in nodes)
    assert "anyOf" not in support["chart"] and support["chart"]["$ref"].endswith("/ChartData")


def test_cycle_schema_has_no_expected_value_on_the_exercise_card(cycle_schema):
    assert "expected_value" not in cycle_schema["$defs"]["ExerciseCard"]["properties"]
    assert "expected_value" not in ExerciseCard.model_fields


def test_cycle_schema_is_a_valid_batch_schema_for_validation():
    parsed = CycleSectionsBatchSchema.model_validate({"sections": [cycle_section()]})

    assert parsed.sections[0].cycle is not None and len(parsed.sections[0].cycle.steps) == 4


# ─── Les autres schémas Gemini restent à l'ancien format ──────────────────────────


@pytest.mark.parametrize("schema", [
    SectionsBatchSchema.model_json_schema(),
    COURSE_GENERATION_GEMINI_SCHEMA,
    CoverageCompletionSchema.model_json_schema(),
    CourseGenerationSchema.model_json_schema(),
])
def test_old_format_schemas_do_not_expose_the_cycle(schema):
    dumped = json.dumps(schema)

    assert "cycle" not in schema["$defs"]["Section"]["properties"]
    assert "widget_brief" not in schema["$defs"]["Section"]["properties"]
    assert "SectionCycle" not in dumped and "StepSupport" not in dumped


def test_old_batch_schema_keeps_its_requirements_and_all_old_fields():
    section = SectionsBatchSchema.model_json_schema()["$defs"]["Section"]

    assert {"subsections", "challenge", "faded_example", "check_questions", "key_takeaways"} <= set(section["required"])
    assert set(OLD_FIELDS) <= set(section["properties"])


# ─── Validation Pydantic : anciens JSON valides ────────────────────────────────────


def test_old_json_without_cycle_stays_valid():
    section = Section.model_validate(old_format_section())

    assert section.cycle is None and section.widget_brief is None
    assert len(section.subsections) == 3


def test_minimal_old_json_is_still_valid():
    assert Section.model_validate({"type": "development", "title": "t"}).cycle is None


def test_pydantic_stays_lenient_about_missing_cycle_parts():
    """Seul le JSON Schema envoyé à Gemini exige les champs : une réponse partielle est signalée, jamais rejetée."""
    section = Section.model_validate({"type": "development", "title": "t", "cycle": {"steps": []}})

    assert section.cycle is not None and section.cycle.challenge.options == []


def test_exercise_card_ignores_a_stray_expected_value():
    card = ExerciseCard.model_validate({"statement": "s", "hint": "h", "steps": ["a"], "result": "r", "expected_value": 4.0})

    assert not hasattr(card, "expected_value")


# ─── API : CourseSection expose le cycle, tolère l'invalide ─────────────────────────────


def _api_section(**extra) -> CourseSection:
    return CourseSection(
        id="0", title="T", quoi="", pourquoi="", comment="",
        worked_example=WorkedExample(statement="", steps=[], result=""), **extra,
    )


def test_course_section_without_cycle_has_none_and_serializes_it():
    section = _api_section()

    assert section.cycle is None and section.widget_brief is None
    assert section.model_dump(mode="json")["cycle"] is None


def test_course_section_exposes_a_mapped_cycle():
    cycle, brief = map_section_cycle(Section.model_validate(cycle_section()), map_question=lambda q: _question_dict(q))
    section = _api_section(cycle=cycle, widget_brief=brief)

    assert section.cycle is not None and section.cycle.challenge.question
    assert section.cycle.steps[1].support.kind == "chart"
    assert section.widget_brief is not None and section.widget_brief.step_number == 2


def test_course_section_drops_an_invalid_cycle_instead_of_failing():
    """Un cycle illisible (ici : défi sans options) ne fait jamais échouer le cours : jamais de 500."""
    section = _api_section(cycle={"challenge": {"question": "q", "options": [], "correct_indices": []}, "steps": []})

    assert section.cycle is None


def test_course_section_drops_an_invalid_widget_brief_instead_of_failing():
    section = _api_section(widget_brief={"step_number": 0, "brief": ""})

    assert section.widget_brief is None


def test_course_section_bounds_text_lengths_of_the_cycle():
    cycle, _ = map_section_cycle(
        Section.model_validate(cycle_section(mutate=lambda s: s["cycle"]["challenge"].update(question="x" * 5000))),
        map_question=_question_dict,
    )

    assert cycle is not None and len(cycle["challenge"]["question"]) <= 400
    # La validation d'API est la dernière barrière : un texte hors borne fait ignorer le cycle, pas le cours.
    cycle["challenge"]["question"] = "x" * 5000
    assert _api_section(cycle=cycle).cycle is None


def _question_dict(q) -> dict:
    from app.services.course_generator import _map_quiz_question

    return _map_quiz_question(q)
