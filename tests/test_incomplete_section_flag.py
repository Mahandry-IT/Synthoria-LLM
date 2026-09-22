from app.api.schemas import ApiPlannedSection
from app.services.course_generator import INCOMPLETE_SECTION_NOTICE, _map_sections_to_course_sections, is_incomplete_section
from app.services.course_plan_generator import _incomplete_section
from app.schemas.course_generation import Section


def _planned(title: str = "Notion", subtopics: list[str] | None = None) -> ApiPlannedSection:
    return ApiPlannedSection(type="development", title=title, objective="Comprendre", subtopics=subtopics or [], order=1)


def test_incomplete_section_fallback_is_flagged_by_is_incomplete_section():
    section = _incomplete_section(_planned())

    assert is_incomplete_section(section) is True


def test_normal_section_is_not_flagged():
    section = Section.model_validate({
        "type": "development", "title": "T", "blocks": [],
        "subsections": [{"title": "Quoi", "blocks": [{"type": "text", "text": "définition normale"}]}],
    })

    assert is_incomplete_section(section) is False


def test_mapping_sets_incomplete_true_on_fallback_section():
    fallback = _incomplete_section(_planned("Notion 1"))

    mapped = _map_sections_to_course_sections([fallback])

    assert len(mapped) == 1 and mapped[0].incomplete is True


def test_mapping_sets_incomplete_false_on_a_normal_section():
    section = Section.model_validate({
        "type": "development", "title": "T", "blocks": [],
        "subsections": [
            {"title": "Pourquoi", "blocks": [{"type": "text", "text": "raison"}]},
            {"title": "Quoi", "blocks": [{"type": "text", "text": "définition"}]},
            {"title": "Comment", "blocks": [{"type": "text", "text": "mécanisme"}]},
        ],
    })

    mapped = _map_sections_to_course_sections([section])

    assert mapped[0].incomplete is False


def test_default_course_section_is_not_incomplete_and_has_no_note():
    from app.api.schemas import CourseSection, WorkedExample

    section = CourseSection(
        id="0", title="T", quoi="q", pourquoi="p", comment="c",
        worked_example=WorkedExample(statement="", steps=[], result=""),
    )

    assert section.incomplete is False and section.note == ""


def test_notice_text_is_shared_between_fallback_and_detection():
    """Empêche une régression où le texte du repli et celui du détecteur divergeraient silencieusement."""
    section = _incomplete_section(_planned())
    assert any(
        b.text == INCOMPLETE_SECTION_NOTICE for sub in section.subsections for b in sub.blocks
    )
