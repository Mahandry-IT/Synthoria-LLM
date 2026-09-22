import pytest
from pydantic import ValidationError

from app.api.schemas import QuizQuestion as ApiQuizQuestion
from app.schemas.course_generation import QuizQuestion

BASE = {
    "question": "Q", "choices": ["A", "B"], "correct_indices": [0], "difficulty": "normale",
    "explanation": "", "requires_calculation": False,
}


def test_llm_section_refs_are_bounded_deduplicated_and_never_blocking():
    q = QuizQuestion.model_validate({**BASE, "section_refs": [3, 1, 3, 0, -2, 9999] + list(range(10, 40))})

    assert q.section_refs == sorted(q.section_refs) and len(q.section_refs) == 10
    assert all(1 <= r <= 500 for r in q.section_refs) and q.section_refs[:3] == [1, 3, 10]


def test_llm_section_refs_default_to_empty():
    assert QuizQuestion.model_validate(BASE).section_refs == []


def test_api_rejects_out_of_range_section_refs():
    api = {"question": "Q", "options": ["A", "B"], "correct_option_indices": [0]}

    assert ApiQuizQuestion.model_validate({**api, "section_refs": [1, 2]}).section_refs == [1, 2]
    for bad in ([0], [501], list(range(1, 12))):
        with pytest.raises(ValidationError):
            ApiQuizQuestion.model_validate({**api, "section_refs": bad})
