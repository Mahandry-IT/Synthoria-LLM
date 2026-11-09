"""Schéma Gemini des lots de sections : le contenu d'une section DEVELOPMENT y est obligatoire."""

from app.schemas.course_generation import CourseGenerationSchema, Section, SectionsBatchSchema


def _defs(schema: dict) -> dict:
    return schema["$defs"]


def test_batch_schema_requires_development_content():
    section = _defs(SectionsBatchSchema.model_json_schema())["Section"]

    for field in ("subsections", "challenge", "challenge_key_points", "faded_example", "check_questions", "recall_prompt"):
        assert field in section["required"]
    assert "covered_subtopics" not in section["required"]  # n'a de sens que depuis un plan
    assert section["properties"]["subsections"]["minItems"] == 3


def test_batch_schema_requires_a_block_in_each_subsection():
    subsection = _defs(SectionsBatchSchema.model_json_schema())["Subsection"]

    assert subsection["properties"]["blocks"]["minItems"] == 1


def test_batch_schema_does_not_leak_into_other_schemas():
    """Introduction et clôture n'ont pas de sous-sections : le schéma du cours reste permissif."""
    section = _defs(CourseGenerationSchema.model_json_schema())["Section"]

    assert "subsections" not in section["required"]
    assert "minItems" not in section["properties"]["subsections"]


def test_pydantic_validation_stays_lenient():
    """Seul le JSON Schema envoyé à Gemini est durci : une section lue en base reste valide."""
    assert Section.model_validate({"type": "development", "title": "t"}).subsections == []
