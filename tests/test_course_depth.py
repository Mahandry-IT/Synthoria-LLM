"""Profils de mode (express / standard / approfondi) : valeurs, repli rétro-compatible, consignes."""

import pytest

from app.core.config import Settings
from app.services.course_depth import (
    COURSE_DEPTHS,
    LEGACY_DEPTH,
    PROFILES,
    depth_of_plan,
    depth_of_session,
    get_profile,
    render_course_rules,
    render_plan_rules,
    render_quiz_rules,
    render_rules,
)


def test_profiles_match_the_reference_table():
    express, standard, deep = PROFILES["express"], PROFILES["standard"], PROFILES["approfondi"]

    assert (express.min_sections, express.max_sections) == (3, 5)
    assert (standard.min_sections, standard.max_sections) == (6, 8)
    assert (deep.min_sections, deep.max_sections) == (6, None)  # piloté par la couverture
    assert {p.non_text_ratio for p in PROFILES.values()} == {0.5}
    assert [p.max_prose_words for p in (express, standard, deep)] == [150, 300, 500]
    assert [p.max_blocks for p in (express, standard, deep)] == [5, 8, 12]
    assert [(p.check_questions_min, p.check_questions_max) for p in (express, standard, deep)] == [
        (1, 2), (2, 3), (2, 3)
    ]
    assert [(p.quiz_min, p.quiz_max, p.quiz_open_ended) for p in (express, standard, deep)] == [
        (5, 6, False), (8, 10, False), (10, 12, True)
    ]


def test_profiles_are_frozen():
    with pytest.raises(AttributeError):
        PROFILES["express"].max_blocks = 99  # type: ignore[misc]


@pytest.mark.parametrize("value", [None, "", "inconnu", 42])
def test_missing_or_unknown_depth_falls_back_to_approfondi(value):
    assert LEGACY_DEPTH == "approfondi"
    assert get_profile(value).name == "approfondi"


def test_depth_of_session_reads_meta_and_defaults_for_legacy_sessions():
    assert depth_of_session({"meta": {"depth": "express"}}) == "express"
    assert depth_of_session({"meta": {"title": "ancien cours"}}) == "approfondi"
    assert depth_of_session({}) == "approfondi"
    assert depth_of_session(None) == "approfondi"


def test_depth_of_plan_defaults_for_rows_without_column():
    class Row:
        depth = "standard"

    assert depth_of_plan(Row()) == "standard"
    assert depth_of_plan(object()) == "approfondi"


def test_render_rules_carries_the_profile_figures():
    rules = render_rules(get_profile("standard"))

    assert "300 mots" in rules and "8 blocs" in rules and "50%" in rules
    assert "2 à 3 `check_questions`" in rules
    assert "la couverture gagne" in rules  # arbitrage documenté
    assert "1 à 2 `check_questions`" in render_rules(get_profile("express"))


def test_render_plan_rules_floor_vs_strict_range():
    assert "entre 3 et 5" in render_plan_rules(get_profile("express"))
    deep = render_plan_rules(get_profile("approfondi"))
    assert "au moins 6" in deep and "jamais un plafond" in deep


def test_render_quiz_rules():
    assert "5 à 6 questions" in render_quiz_rules(get_profile("express"))
    assert "au moins 10 à 12" in render_quiz_rules(get_profile("approfondi"))


def test_render_course_rules_combines_everything():
    rules = render_course_rules(get_profile("express"))
    assert "entre 3 et 5" in rules and "150 mots" in rules and "5 à 6 questions" in rules


def test_default_course_depth_setting():
    assert Settings(gemini_api_key="k").default_course_depth == "approfondi"
    assert Settings(gemini_api_key="k", default_course_depth="express").default_course_depth == "express"
    assert set(COURSE_DEPTHS) == {"express", "standard", "approfondi"}
