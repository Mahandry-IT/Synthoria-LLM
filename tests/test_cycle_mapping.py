"""Format « cycle » : mapping Gemini → API (éléments écartés, mélange avec index recalculés, jamais de 500)."""

import random
from collections.abc import Callable
from typing import Any

import pytest

from app.schemas.course_generation import Section
from app.services.course_generator import (
    _map_quiz_question,
    _map_sections_to_course_sections,
    is_incomplete_section_dict,
)
from app.services.cycle_mapping import cycle_legacy_text, map_section_cycle
from tests.cycle_fixtures import cycle_section, old_format_section

SEEDS = range(40)
RIGHT_CHALLENGE = {"Chaque part a 64 adresses", "Les parts couvrent tout le réseau"}


def _map(mutate: Callable[[dict[str, Any]], None] | None = None, seed: int = 0):
    section = Section.model_validate(cycle_section(mutate=mutate))
    return map_section_cycle(section, map_question=_map_quiz_question, rng=random.Random(seed))


def _cycle(mutate: Callable[[dict[str, Any]], None] | None = None, seed: int = 0) -> dict[str, Any]:
    cycle, _ = _map(mutate, seed)
    assert cycle is not None
    return cycle


# ─── Cas nominal ───────────────────────────────────────────────────────────────────


def test_mapping_keeps_the_five_moments_and_the_widget_brief():
    cycle, brief = _map()

    assert cycle is not None
    assert [s["support"]["kind"] for s in cycle["steps"]] == ["classify", "chart", "fill_table", "order"]
    assert len(cycle["vocabulary"]) == 3 and len(cycle["traps"]["entries"]) == 2
    assert cycle["trial"]["kind"] == "qcm" and cycle["thread"]
    assert brief == {"step_number": 2, "brief": "Curseur du nombre de parts : la taille de chaque part se met à jour."}


def test_section_without_cycle_maps_to_none():
    assert map_section_cycle(Section.model_validate(old_format_section()), map_question=_map_quiz_question) == (None, None)


def test_course_section_mapping_carries_the_cycle_and_is_not_incomplete():
    mapped = _map_sections_to_course_sections([Section.model_validate(cycle_section())], rng=random.Random(1))

    assert len(mapped) == 1
    section = mapped[0]
    assert section.cycle is not None and section.widget_brief is not None
    assert section.incomplete is False
    assert section.key_takeaways and section.recall_prompt is not None
    assert section.pitfalls_to_avoid == [] and section.check_questions == [] and section.challenge == ""


def test_legacy_text_fields_are_filled_for_podcast_assistant_and_detection():
    section = _map_sections_to_course_sections([Section.model_validate(cycle_section())])[0]
    pourquoi, quoi, comment = cycle_legacy_text(section.cycle.model_dump(mode="json"))

    assert section.pourquoi == pourquoi and "192.168.1.0/24" in section.pourquoi
    assert "Masque :" in section.quoi
    assert "Un réseau /24" in section.comment
    assert is_incomplete_section_dict(section.model_dump(mode="json")) is False


def test_cycle_is_only_mapped_on_development_sections():
    """RG-1 : l'introduction et le résumé n'ont jamais de cycle, même si le modèle en renvoyait un."""
    intro = cycle_section(mutate=lambda s: s.update(type="introduction"))
    intro["blocks"] = [{"type": "text", "text": "Bienvenue"}]

    mapped = _map_sections_to_course_sections([Section.model_validate(intro)])

    assert mapped[0].cycle is None and mapped[0].widget_brief is None


def test_old_format_section_in_the_same_batch_is_untouched():
    mapped = _map_sections_to_course_sections(
        [Section.model_validate(old_format_section("A")), Section.model_validate(cycle_section("B"))]
    )

    assert mapped[0].cycle is None and mapped[0].incomplete is False and mapped[0].subsections
    assert mapped[1].cycle is not None


# ─── Mélange et index recalculés ────────────────────────────────────────────────────


def test_challenge_options_are_shuffled_and_correct_indices_recomputed():
    positions: set[tuple[int, ...]] = set()
    for seed in SEEDS:
        challenge = _cycle(seed=seed)["challenge"]
        assert {challenge["options"][i] for i in challenge["correct_indices"]} == RIGHT_CHALLENGE
        assert sorted(challenge["options"]) == sorted(cycle_section()["cycle"]["challenge"]["options"])
        positions.add(tuple(challenge["correct_indices"]))
    assert len(positions) > 3  # la place des bonnes réponses ne dépend plus de l'ordre d'écriture


def test_prediction_choice_options_are_shuffled_and_correct_index_recomputed():
    indices = set()
    for seed in SEEDS:
        prediction = _cycle(seed=seed)["steps"][1]["prediction"]
        assert prediction["options"][prediction["correct_index"]] == "64"
        indices.add(prediction["correct_index"])
    assert indices == {0, 1, 2}


def test_trap_statements_are_shuffled_and_wrong_index_recomputed():
    wrong_positions = set()
    for seed in SEEDS:
        first, second = _cycle(seed=seed)["traps"]["entries"]
        assert first["statements"][first["wrong_index"]] == "Un /26 compte 26 adresses"
        assert second["statements"][second["wrong_index"]] == "Un sous-réseau peut dépasser le réseau"
        wrong_positions.add(first["wrong_index"])
    assert wrong_positions == {0, 1, 2}


def test_spot_rewrite_options_are_shuffled_with_their_own_index():
    def to_rewrite(section):
        section["cycle"]["traps"]["mode"] = "spot_rewrite"
        for entry in section["cycle"]["traps"]["entries"]:
            entry["rewrite_options"] = ["Mauvaise", "Bonne", "Autre"]
            entry["correct_rewrite_index"] = 1

    seen = set()
    for seed in SEEDS:
        for entry in _cycle(to_rewrite, seed)["traps"]["entries"]:
            assert entry["rewrite_options"][entry["correct_rewrite_index"]] == "Bonne"
            seen.add(entry["correct_rewrite_index"])
    assert seen == {0, 1, 2}


def test_trial_question_choices_are_shuffled_with_their_explanations_aligned():
    seen = set()
    for seed in SEEDS:
        question = _cycle(seed=seed)["trial"]["questions"][0]
        (right,) = question["correct_option_indices"]
        assert question["options"][right] == "64"
        assert question["explanation_per_choice"][right] == "Exact."
        seen.add(right)
    assert seen == {0, 1, 2}


def test_trial_question_with_misaligned_per_choice_feedback_drops_only_the_feedback():
    cycle = _cycle(lambda s: s["cycle"]["trial"]["questions"][0].update(explanation_per_choice=["un seul"]))

    assert cycle["trial"]["questions"][0]["explanation_per_choice"] == []


def test_classify_cases_are_shuffled_and_keep_their_category():
    orders = set()
    for seed in SEEDS:
        support = _cycle(seed=seed)["steps"][0]["support"]
        by_label = {c["label"]: support["categories"][c["category_index"]] for c in support["cases"]}
        assert by_label == {"192.168.1.10": "Sous-réseau A", "192.168.1.70": "Sous-réseau B", "192.168.1.20": "Sous-réseau A"}
        orders.add(tuple(c["label"] for c in support["cases"]))
    assert len(orders) > 1


def test_order_items_keep_the_correct_order_for_the_client_to_shuffle():
    support = _cycle()["steps"][3]["support"]

    assert support["order_items"] == ["Compter les adresses", "Diviser par quatre", "Lister les débuts"]


# ─── Éléments invalides ou vides écartés ─────────────────────────────────────────────


def test_blank_and_duplicate_challenge_options_are_dropped_without_losing_the_right_answers():
    def mutate(section):
        challenge = section["cycle"]["challenge"]
        challenge["options"] = ["Faux", "  ", "Vrai", "vrai", "Autre faux"]
        challenge["correct_indices"] = [2, 3, 9]  # 3 est un doublon, 9 n'existe pas

    for seed in SEEDS:
        challenge = _cycle(mutate, seed)["challenge"]
        assert sorted(challenge["options"]) == ["Autre faux", "Faux", "Vrai"]
        assert [challenge["options"][i] for i in challenge["correct_indices"]] == ["Vrai"]


def test_challenge_without_a_right_answer_drops_the_whole_cycle_not_the_course():
    cycle, brief = _map(lambda s: s["cycle"]["challenge"].update(correct_indices=[7]))

    assert (cycle, brief) == (None, None)


def test_section_whose_cycle_is_unusable_becomes_an_incomplete_fallback_never_disappears():
    broken = cycle_section(mutate=lambda s: s["cycle"].update(steps=[]))

    mapped = _map_sections_to_course_sections([Section.model_validate(broken)])

    assert len(mapped) == 1 and mapped[0].cycle is None
    assert mapped[0].incomplete is True and mapped[0].title == "Sous-réseaux"
    assert is_incomplete_section_dict(mapped[0].model_dump(mode="json"))


def test_step_with_an_empty_table_is_dropped_and_widget_brief_is_renumbered():
    def mutate(section):
        section["cycle"]["steps"][0]["support"] = {"kind": "table", "caption": "vide", "table": {"headers": ["a"], "rows": []}}
        section["widget_brief"]["step_number"] = 2  # le graphique, désormais à la place 1

    cycle, brief = _map(mutate)

    assert cycle is not None and [s["support"]["kind"] for s in cycle["steps"]] == ["chart", "fill_table", "order"]
    assert brief["step_number"] == 1


def test_widget_brief_pointing_at_a_dropped_step_is_removed():
    def mutate(section):
        section["cycle"]["steps"][1]["support"] = {"kind": "diagram", "caption": "d", "diagram": {"kind": "flowchart", "mermaid": "  "}}

    cycle, brief = _map(mutate)

    assert cycle is not None and len(cycle["steps"]) == 3 and brief is None


def test_widget_brief_beyond_the_steps_is_removed():
    assert _map(lambda s: s["widget_brief"].update(step_number=12))[1] is None


def test_step_without_text_is_dropped():
    cycle = _cycle(lambda s: s["cycle"]["steps"][2].update(text="   "))

    assert len(cycle["steps"]) == 3


def test_prediction_with_an_invalid_index_is_dropped_but_the_step_is_kept():
    cycle = _cycle(lambda s: s["cycle"]["steps"][1]["prediction"].update(correct_index=5))

    assert "prediction" not in cycle["steps"][1] and cycle["steps"][1]["support"]["kind"] == "chart"


def test_prediction_value_tolerance_is_capped_at_ten_percent():
    def mutate(section):
        section["cycle"]["steps"][1]["prediction"] = {
            "kind": "value", "question": "Combien ?", "expected_value": 200.5, "tolerance": 100.0, "unit": "adresses", "outcome": "En réalité 200,5.",
        }

    prediction = _cycle(mutate)["steps"][1]["prediction"]

    assert prediction["kind"] == "value" and prediction["tolerance"] == pytest.approx(20.05)
    assert prediction["expected_value"] == 200.5 and prediction["unit"] == "adresses"


@pytest.mark.parametrize("tolerance", [None, -3.0])
def test_prediction_value_without_a_usable_tolerance_gets_zero(tolerance):
    def mutate(section):
        section["cycle"]["steps"][1]["prediction"] = {
            "kind": "value", "question": "Combien ?", "expected_value": 62.0, "tolerance": tolerance, "outcome": "En réalité 62.",
        }

    assert _cycle(mutate)["steps"][1]["prediction"]["tolerance"] == 0.0


def test_prediction_value_without_an_expected_value_is_dropped():
    def mutate(section):
        section["cycle"]["steps"][1]["prediction"] = {"kind": "value", "question": "Combien ?", "outcome": "En réalité."}

    assert "prediction" not in _cycle(mutate)["steps"][1]


def test_fill_table_with_no_valid_blank_degrades_to_a_plain_table():
    def mutate(section):
        section["cycle"]["steps"][2]["support"]["blanks"] = [{"row": 9, "col": 0}, {"row": 0, "col": 7}]

    support = _cycle(mutate)["steps"][2]["support"]

    assert support["kind"] == "table" and not support["blanks"] and support["table"]["rows"]


def test_fill_table_distractors_never_repeat_an_answer():
    def mutate(section):
        section["cycle"]["steps"][2]["support"]["distractors"] = ["64", "65", "65", "70", "", "71", "72"]

    support = _cycle(mutate)["steps"][2]["support"]

    assert support["distractors"] == ["65", "70", "71"]


def test_classify_with_too_few_cases_drops_the_step():
    def mutate(section):
        section["cycle"]["steps"][0]["support"]["cases"] = section["cycle"]["steps"][0]["support"]["cases"][:1]

    assert len(_cycle(mutate)["steps"]) == 3


def test_classify_case_pointing_at_a_blank_category_is_dropped_and_indices_follow():
    def mutate(section):
        support = section["cycle"]["steps"][0]["support"]
        support["categories"] = ["", "Sous-réseau A", "Sous-réseau B"]
        support["cases"] = [
            {"label": "a", "category_index": 1}, {"label": "b", "category_index": 2},
            {"label": "c", "category_index": 0}, {"label": "d", "category_index": 1},
        ]

    support = _cycle(mutate)["steps"][0]["support"]

    assert support["categories"] == ["Sous-réseau A", "Sous-réseau B"]
    assert {c["label"]: support["categories"][c["category_index"]] for c in support["cases"]} == {
        "a": "Sous-réseau A", "b": "Sous-réseau B", "d": "Sous-réseau A",
    }


def test_match_pairs_with_a_repeated_right_value_keep_the_first_only():
    def mutate(section):
        section["cycle"]["steps"][3]["support"] = {
            "kind": "match", "caption": "m",
            "match_pairs": [{"left": "a", "right": "x"}, {"left": "b", "right": "X"}, {"left": "c", "right": "y"}],
        }

    pairs = _cycle(mutate)["steps"][3]["support"]["match_pairs"]

    assert pairs == [{"left": "a", "right": "x"}, {"left": "c", "right": "y"}]


def test_chart_with_series_shorter_than_labels_is_cut_to_the_shortest():
    def mutate(section):
        section["cycle"]["steps"][1]["support"]["chart"]["series"][0]["values"] = [64.0, 64.0, 64.0]

    chart = _cycle(mutate)["steps"][1]["support"]["chart"]

    assert chart["labels"] == ["A", "B", "C"] and chart["series"][0]["values"] == [64.0, 64.0, 64.0]


def test_chart_with_non_finite_values_is_dropped_with_its_step():
    def mutate(section):
        section["cycle"]["steps"][1]["support"]["chart"]["series"][0]["values"][0] = float("nan")

    assert len(_cycle(mutate)["steps"]) == 3


def test_trap_entry_with_an_invalid_index_is_dropped_and_all_invalid_means_no_traps():
    def first_invalid(section):
        section["cycle"]["traps"]["entries"][0]["wrong_index"] = 8

    def all_invalid(section):
        for entry in section["cycle"]["traps"]["entries"]:
            entry["wrong_index"] = 8

    assert len(_cycle(first_invalid)["traps"]["entries"]) == 1
    assert "traps" not in _cycle(all_invalid)


def test_cards_traps_use_the_claim_and_fall_back_on_the_false_statement():
    def mutate(section):
        section["cycle"]["traps"]["mode"] = "cards"
        section["cycle"]["traps"]["entries"][0]["claim"] = "Le /26 compte 26 adresses"
        # entrée 2 : pas de `claim`, la fausse affirmation (`wrong_index`) en tient lieu

    entries = _cycle(mutate)["traps"]["entries"]

    assert [e["claim"] for e in entries] == ["Le /26 compte 26 adresses", "Un sous-réseau peut dépasser le réseau"]
    assert all(not e.get("statements") for e in entries)


def test_vocabulary_drops_empty_and_repeated_terms():
    def mutate(section):
        section["cycle"]["vocabulary"] += [{"term": "", "plain": "vide"}, {"term": "Masque", "plain": "doublon"}, {"term": "Hôte", "plain": ""}]

    assert [v["term"] for v in _cycle(mutate)["vocabulary"]] == ["Masque", "Sous-réseau", "Plage"]


def test_exercise_card_trial_carries_no_expected_value():
    def mutate(section):
        section["cycle"]["trial"] = {
            "kind": "exercise_card",
            "card": {"statement": "Découpe 10.0.0.0/24 en 2.", "hint": "Divise 256.", "steps": ["256 / 2 = 128", "Deux parts de 128"], "result": "128 adresses", "expected_value": 128.0},
        }

    trial = _cycle(mutate)["trial"]

    assert trial["kind"] == "exercise_card" and "expected_value" not in trial["card"]
    assert trial["card"]["steps"] == ["256 / 2 = 128", "Deux parts de 128"]


def test_trial_falls_back_on_the_card_when_the_qcm_is_unusable_and_conversely():
    def card_but_empty_qcm(section):
        section["cycle"]["trial"] = {"kind": "qcm", "questions": [], "card": {"statement": "s", "hint": "", "steps": ["a"], "result": ""}}

    def qcm_but_empty_card(section):
        section["cycle"]["trial"]["kind"] = "exercise_card"

    assert _cycle(card_but_empty_qcm)["trial"]["kind"] == "exercise_card"
    assert _cycle(qcm_but_empty_card)["trial"]["kind"] == "qcm"


def test_numeric_trial_question_keeps_its_expected_value_and_has_no_options():
    def mutate(section):
        section["cycle"]["trial"]["questions"] = [{
            "question": "Combien d'hôtes ?", "kind": "numeric", "expected_value": 62.0, "unit": "hôtes",
            "difficulty": "normale", "explanation": "e", "requires_calculation": True,
        }]

    question = _cycle(mutate)["trial"]["questions"][0]

    assert question["kind"] == "numeric" and question["expected_value"] == 62.0 and question["options"] == []
    assert question["time_limit_seconds"] == 80  # règles de la réponse numérique existantes


def test_matching_questions_are_not_allowed_in_the_trial():
    def mutate(section):
        section["cycle"]["trial"]["questions"] = [{
            "question": "Associe", "kind": "matching", "choices": ["a", "b", "c"], "subjects": ["1", "2", "3"],
            "correct_matches": [0, 1, 2], "difficulty": "normale", "explanation": "e", "requires_calculation": False,
        }]

    assert "trial" not in _cycle(mutate)


def test_long_texts_are_truncated_to_the_api_bounds():
    def mutate(section):
        section["cycle"]["thread"] = "mot " * 200
        section["cycle"]["steps"][0]["text"] = "phrase très longue. " * 100

    cycle = _cycle(mutate)

    assert len(cycle["thread"]) <= 200 and len(cycle["steps"][0]["text"]) <= 500


def test_mapping_survives_garbage_without_raising():
    def mutate(section):
        section["cycle"]["challenge"]["options"] = ["a", "b", "c"]
        section["cycle"]["steps"][0]["support"]["cases"] = []
        section["cycle"]["steps"][1]["support"]["chart"]["labels"] = []
        section["cycle"]["steps"][2]["support"]["table"] = None
        section["cycle"]["steps"][3]["support"]["order_items"] = []

    assert _map(mutate) == (None, None)  # plus aucune mini-étape exploitable
