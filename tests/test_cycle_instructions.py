"""Consignes de génération : le cycle est décrit sans chiffres de mode, l'ancien format reste décrit."""

import re

import pytest

from app.services.course_generator import _get_teacher_instructions


@pytest.fixture(scope="module")
def instructions() -> str:
    return _get_teacher_instructions()


@pytest.fixture(scope="module")
def cycle_part(instructions: str) -> str:
    return instructions.split("## Cycle format", 1)[1].split("## Language", 1)[0]


def test_both_formats_are_described(instructions):
    assert "### Old format (sections without `cycle`)" in instructions
    assert "**Challenge → Pourquoi → Quoi → Comment → À toi → Vérifie → Explique avec tes mots → Retenir → Éviter**" in instructions
    assert "Subsections of a DEVELOPMENT section in the old format come in this order" in instructions
    assert "The introduction, the summary, the pitfalls and the next steps sections are always written in the old way" in instructions


def test_the_five_moments_and_their_fields_are_described(cycle_part):
    for needle in ("**Défi**", "**Sous le capot**", "**Le mot juste**", "**Les pièges classiques**", "**Épreuve**",
                   "`cycle.challenge`", "`cycle.steps`", "`cycle.vocabulary`", "`cycle.traps`", "`cycle.trial`"):
        assert needle in cycle_part
    assert "NEVER say how many answers are right" in cycle_part


def test_form_rules_are_stated(cycle_part):
    assert "never above 80" in cycle_part and "ONE idea per sentence" in cycle_part
    assert "Never the same support kind in two consecutive steps" in cycle_part
    assert "One formula per step at most" in cycle_part
    assert "A section is never made of text alone" in cycle_part


def test_prediction_rules_are_stated(cycle_part):
    assert "**Never give the answer away.**" in cycle_part and "nor in the `text` of any earlier step" in cycle_part
    assert "at least ONE prediction of the section MUST be `value`" in cycle_part


def test_widget_brief_rule_is_stated(cycle_part):
    assert "vary continuously" in cycle_part
    assert "Law, legal reasoning, grammar, vocabulary and purely procedural subjects" in cycle_part
    assert "`widget_brief` MUST be null there" in cycle_part


def test_the_position_of_a_right_answer_is_never_imposed(cycle_part):
    assert "Never impose or hint at the position of a right answer" in cycle_part
    assert not re.search(r"\b(index|position)\s+(0|1|2)\b|VARY the position|not always index", cycle_part, re.I)


def test_the_exercise_card_has_no_expected_value(cycle_part):
    assert "no expected value" in cycle_part


def test_the_cycle_part_carries_no_mode_figures(cycle_part):
    """Les nombres propres à un mode (mini-étapes, prédictions, pièges, vocabulaire) vivent dans course_depth."""
    assert not re.search(r"\b(express|standard|approfondi)\b", cycle_part, re.I)
    assert not re.search(r"\b\d+ (?:to|à) \d+ (?:mini-steps|steps|predictions|terms|entries|traps)\b", cycle_part)
    assert "in the number given by the course mode rules" in cycle_part
