from app.services.course_generator import _map_sections_to_course_sections
from app.schemas.course_generation import Section


def _cycle_section() -> Section:
    check = {
        "question": "Q ?", "choices": ["A", "B", "C"], "correct_indices": [1], "difficulty": "facile",
        "explanation": "Parce que B.", "explanation_per_choice": ["non", "oui", "non"],
        "requires_calculation": False,
    }
    return Section.model_validate({
        "type": "development", "title": "T", "blocks": [],
        "challenge": "Que se passe-t-il si on double les spires ?",
        "subsections": [
            {"title": "Pourquoi", "blocks": [{"type": "text", "text": "raison"}]},
            {"title": "Quoi", "blocks": [{"type": "text", "text": "definition"}]},
            {"title": "Comment", "blocks": [{"type": "text", "text": "mecanisme"}]},
        ],
        "faded_example": {"statement": "s", "given_steps": ["1"], "hidden_steps": ["2", "3"], "result": "r"},
        "check_questions": [check],
        "recall_prompt": {"prompt": "Explique.", "expected_key_points": ["a", "b"]},
    })


def test_pourquoi_quoi_comment_order_maps_to_right_fields():
    api = _map_sections_to_course_sections([_cycle_section()])[0]

    assert (api.pourquoi, api.quoi, api.comment) == ("raison", "definition", "mecanisme")
    assert [s.title for s in api.subsections] == ["Pourquoi", "Quoi", "Comment"]


def test_cycle_fields_are_exposed():
    api = _map_sections_to_course_sections([_cycle_section()])[0]

    assert api.challenge.startswith("Que se passe-t-il")
    assert api.faded_example.hidden_steps == ["2", "3"]
    assert api.check_questions[0].explanation_per_choice == ["non", "oui", "non"]
    assert api.check_questions[0].correct_option_indices == [1]
    assert api.recall_prompt.expected_key_points == ["a", "b"]


def test_section_without_cycle_fields_stays_valid():
    legacy = Section.model_validate({
        "type": "development", "title": "T", "blocks": [],
        "subsections": [{"title": "Quoi", "blocks": [{"type": "text", "text": "x"}]}],
    })
    api = _map_sections_to_course_sections([legacy])[0]

    assert api.challenge == "" and api.faded_example is None and api.check_questions == [] and api.recall_prompt is None
