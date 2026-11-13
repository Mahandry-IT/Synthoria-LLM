"""Lot 2 : encadrés « retenir » / « éviter » (schéma, anciens cours, complétude) et repli vidéo."""

import pytest

from app.api.schemas import ApiPlannedSection, CourseSection, CourseVideo, WorkedExample
from app.core.config import Settings
from app.schemas.course_generation import CourseGenerationSchema, Section, SectionsBatchSchema
from app.services.course_depth import PROFILES, render_rules
from app.services.course_generator import (
    _map_sections_to_course_sections,
    _truncate,
    is_incomplete_section,
    is_incomplete_section_dict,
)
from app.services.course_plan_generator import _generate_batch, _structural_gaps, section_issues
from app.services.course_videos import assign_sections_by_topic
from tests.lesson_box_fixtures import BOXES_FIELDS

_PITFALL = BOXES_FIELDS["pitfalls_to_avoid"][0]


_LIST_BLOCK = {"type": "list", "list_items": ["a"]}  # évite le problème soft « aucun bloc visuel »


def _section(**extra) -> Section:
    return Section.model_validate({"type": "development", "title": "T", "subsections": [
        {"title": t, "blocks": [{"type": "text", "text": "x"}] + ([_LIST_BLOCK] if t == "Comment" else [])}
        for t in ("Pourquoi", "Quoi", "Comment")
    ], **extra})


# ─── Schéma ──────────────────────────────────────────────────────────────────


def test_batch_schema_requires_the_two_boxes():
    section = SectionsBatchSchema.model_json_schema()["$defs"]["Section"]
    assert {"key_takeaways", "pitfalls_to_avoid"} <= set(section["required"])
    assert section["properties"]["key_takeaways"]["minItems"] == 1
    assert section["properties"]["pitfalls_to_avoid"]["minItems"] == 1


def test_course_schema_keeps_the_boxes_optional():
    section = CourseGenerationSchema.model_json_schema()["$defs"]["Section"]
    assert "key_takeaways" not in section["required"] and "pitfalls_to_avoid" not in section["required"]


def test_old_section_json_without_boxes_still_validates():
    section = _section()
    assert section.key_takeaways == [] and section.pitfalls_to_avoid == []


def test_api_course_section_defaults_to_empty_boxes_for_old_courses():
    api = CourseSection(id="0", title="T", quoi="q", pourquoi="p", comment="c", worked_example=WorkedExample(statement="", steps=[], result=""))
    assert api.key_takeaways == [] and api.pitfalls_to_avoid == []


def test_mapping_bounds_and_cleans_model_text():
    section = _section(key_takeaways=["  a  ", " ", "b" * 500], pitfalls_to_avoid=[
        _PITFALL, {"mistake": "sans remède", "how_to_avoid": " "}, {"mistake": "m" * 900, "how_to_avoid": "h"},
    ])
    [api] = _map_sections_to_course_sections([section])
    assert api.key_takeaways[0] == "a" and len(api.key_takeaways) == 2 and len(api.key_takeaways[1]) == 200
    assert len(api.pitfalls_to_avoid) == 2 and len(api.pitfalls_to_avoid[1].mistake) == 300


def test_truncate_cuts_at_word_boundary_with_ellipsis():
    assert _truncate("un deux trois quatre", 12) == "un deux…"
    assert _truncate("court", 12) == "court"
    assert _truncate("x" * 20, 10) == "x" * 9 + "…"  # aucun espace : coupe franche
    assert len(_truncate("mot " * 100, 50)) <= 50


# ─── Complétude ──────────────────────────────────────────────────────────────


def test_new_section_without_boxes_is_reported_but_never_blocking():
    issues = section_issues(_section(), None)
    assert any("key_takeaways" in i and "pitfalls_to_avoid" in i for i in issues.boxes)
    assert issues.blocking == () and issues.blocking_count == 0
    assert not issues  # à eux seuls, les encadrés ne déclenchent aucune réparation
    assert _structural_gaps(_section()) == []  # sous-sections : inchangé


def test_new_section_with_both_boxes_has_no_box_issue():
    issues = section_issues(_section(**BOXES_FIELDS), None)
    assert issues.boxes == ()


def test_missing_only_pitfalls_is_flagged():
    issues = section_issues(_section(key_takeaways=["a", "b"]), None)
    assert any("pitfalls_to_avoid" in i and "key_takeaways" not in i for i in issues.boxes)


@pytest.mark.asyncio
async def test_batch_keeps_otherwise_valid_section_without_boxes_with_no_extra_call():
    from unittest.mock import AsyncMock

    # 4 blocs dont 2 non textuels : conforme aux règles de forme, seuls les encadrés manquent.
    section = Section.model_validate({"type": "development", "title": "S", "subsections": [
        {"title": "Pourquoi", "blocks": [_LIST_BLOCK]},
        {"title": "Quoi", "blocks": [{"type": "text", "text": "x"}]},
        {"title": "Comment", "blocks": [{"type": "text", "text": "x"}, _LIST_BLOCK]},
    ]})
    assert not section_issues(section, None)  # valide hors encadrés
    gemini = AsyncMock()
    gemini.format_structured.return_value = {"sections": [section.model_dump(mode="json")]}
    planned = ApiPlannedSection(type="development", title="S", objective="", subtopics=[], order=1)

    [result] = await _generate_batch(
        [planned], question="q", mode="question_only", context_block="c", outline="o", gemini_client=gemini
    )

    assert gemini.format_structured.await_count == 1  # aucun appel de réparation
    assert not is_incomplete_section(result)
    assert result.subsections == section.subsections  # gardée telle quelle


def test_old_stored_course_without_boxes_is_not_incomplete():
    stored = {"title": "T", "subsections": [
        {"title": "Pourquoi", "blocks": [{"type": "text", "text": "p"}]},
        {"title": "Quoi", "blocks": [{"type": "text", "text": "q"}]},
        {"title": "Comment", "blocks": [{"type": "text", "text": "c"}]},
    ]}
    assert is_incomplete_section_dict(stored) is False


# ─── Bornes par mode ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("mode, takeaways, pitfalls", [("express", "2", "1"), ("standard", "3", "1"), ("approfondi", "3 à 4", "2")])
def test_rules_inject_box_bounds_per_mode(mode, takeaways, pitfalls):
    profile = PROFILES[mode]
    assert f"- {takeaways} `key_takeaways`" in render_rules(profile)
    assert f"et {pitfalls} `pitfalls_to_avoid`" in render_rules(profile)


# ─── Repli vidéo par sujet ───────────────────────────────────────────────────


def _video(vid: str, title: str, **extra) -> CourseVideo:
    return CourseVideo(
        video_id=vid, url=f"https://www.youtube.com/watch?v={vid}", embed_url=f"https://www.youtube.com/embed/{vid}",
        thumbnail_url="t", title=title, **extra,
    )


_SECTIONS = [
    ("0", "Les probabilités conditionnelles"), ("1", "Loi normale"), ("2", "Variance et écart-type"),
    ("3", "Introduction"),
]


def test_fallback_attaches_video_to_matching_section():
    [v] = assign_sections_by_topic([_video("a", "Probabilités conditionnelles expliquées")], _SECTIONS, Settings())
    assert v.section_id == "0"


def test_fallback_leaves_unrelated_video_at_course_level():
    [v] = assign_sections_by_topic([_video("a", "Histoire de la statistique")], _SECTIONS, Settings())
    assert v.section_id is None


def test_fallback_never_attaches_on_a_generic_section_title():
    [v] = assign_sections_by_topic([_video("a", "Introduction à la statistique")], _SECTIONS, Settings())
    assert v.section_id is None


def test_fallback_needs_two_shared_words_for_a_long_section_title():
    [one] = assign_sections_by_topic([_video("a", "Variance expliquée simplement")], _SECTIONS, Settings())
    [two] = assign_sections_by_topic([_video("b", "Variance et écart-type en pratique")], _SECTIONS, Settings())
    assert one.section_id is None and two.section_id == "2"


def test_fallback_ignores_ranked_videos_even_with_null_section():
    ranked = _video("a", "Probabilités conditionnelles", category="cours")
    [v] = assign_sections_by_topic([ranked], _SECTIONS, Settings())
    assert v.section_id is None


def test_fallback_keeps_existing_section_id():
    [v] = assign_sections_by_topic([_video("a", "Loi normale", section_id="2")], _SECTIONS, Settings())
    assert v.section_id == "2"


def test_fallback_respects_per_section_cap():
    settings = Settings(course_videos_max_per_section=1)
    videos = [_video("a", "Loi normale cours"), _video("b", "Loi normale exercices")]
    result = assign_sections_by_topic(videos, _SECTIONS, settings)
    assert [v.section_id for v in result] == ["1", None]


def test_fallback_ambiguous_match_stays_unattached():
    sections = [("0", "Variance empirique"), ("1", "Variance théorique")]
    [v] = assign_sections_by_topic([_video("a", "Variance")], sections, Settings())
    assert v.section_id is None


# ─── Réparation ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_regenerator_repair_asks_again_for_missing_boxes():
    """Réparation déclenchée par un autre motif (sous-section vide) : la 2e requête réclame aussi les encadrés."""
    from unittest.mock import AsyncMock

    from app.services.section_regenerator import _repair

    complete = _section(**BOXES_FIELDS).model_dump(mode="json")
    gemini = AsyncMock()
    gemini.format_structured.return_value = {"sections": [complete]}
    broken = Section.model_validate({"type": "development", "title": "T", "subsections": [
        {"title": "Pourquoi", "blocks": [{"type": "text", "text": "x"}]},
        {"title": "Quoi", "blocks": [{"type": "text", "text": "x"}]},
        {"title": "Comment", "blocks": []},
    ]})
    issues = section_issues(broken, None, PROFILES["standard"])
    assert issues.blocking and issues.boxes

    result = await _repair(
        broken, "base", profile=PROFILES["standard"], system_instruction="s", gemini_client=gemini, max_repairs=1,
    )

    prompt = gemini.format_structured.await_args.kwargs["raw_answer"]
    assert issues.blocking[0] in prompt and issues.boxes[0] in prompt
    assert result.key_takeaways == ["à retenir"]
