"""Corrections de la code review du lot 1 (format « cycle », mode standard) : une section ne disparaît jamais,
reconstruction toujours exigée, champs omis tolérés, maxima tronqués (soft), épreuve et défi bornés, ajout de section
validé, nits."""

import random
from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from app.api.schemas import (
    CYCLE_OPTION_MAX,
    CYCLE_QUESTION_MAX,
    ApiCycleChallenge,
    ApiCycleTrial,
    ApiPlannedSection,
)
from app.core.config import Settings
from app.schemas.course_generation import (
    CycleSectionsBatchSchema,
    Section,
    StepPrediction,
    StepSupport,
    SupportKind,
)
from app.services.course_depth import get_profile
from app.services.course_generator import (
    _map_quiz_question,
    _map_sections_to_course_sections,
    is_incomplete_section,
    is_incomplete_section_dict,
)
from app.services.course_plan_generator import _finalize_incomplete, _structural_gaps, section_issues
from app.services.course_section_adder import add_course_sections
from app.services.cycle_mapping import cycle_gaps, map_section_cycle
from app.services.cycle_validation import _numbers, cycle_bounds_issues, cycle_structural_gaps
from app.services.visual_validation import _ANNOUNCES_COUNT_RE, cycle_form_issues
from tests.cycle_fixtures import cycle_section
from tests.test_cycle_generation import _dev, _fake_gemini, _generate
from tests.test_cycle_regeneration import _gemini, _row, _store

STANDARD = get_profile("standard")
Mutate = Callable[[dict[str, Any]], None]


def _section(mutate: Mutate | None = None) -> Section:
    return Section.model_validate(cycle_section(mutate=mutate))


def _steps(section: dict[str, Any]) -> list[dict[str, Any]]:
    return section["cycle"]["steps"]


def _gaps(mutate: Mutate | None = None) -> list[str]:
    return cycle_gaps(_section(mutate), STANDARD, _map_quiz_question)


def _mapped(mutate: Mutate | None = None, seed: int = 0) -> dict[str, Any]:
    cycle, _ = map_section_cycle(_section(mutate), map_question=_map_quiz_question, rng=random.Random(seed))
    assert cycle is not None
    return cycle


def _has(items: list[str], *words: str) -> bool:
    return any(all(w in item for w in words) for item in items)


def _table_support(caption: str = "t") -> dict[str, Any]:
    return {"kind": "table", "caption": caption, "table": {"headers": ["a", "b", "c"], "rows": [["1", "2", "3"]]}}


# ─── 1. Une section ne disparaît jamais sans être marquée incomplète ────────────────────────────────


def _duplicate_options_challenge(section: dict[str, Any]) -> None:
    section["cycle"]["challenge"].update(options=["a", "A", "a"], correct_indices=[0])


def test_challenge_with_duplicate_options_is_a_blocking_gap_not_a_silent_loss():
    """Sonde de la review : options ["a", "A", "a"], correct_indices [0] : 0 écart et 0 section."""
    section = _section(_duplicate_options_challenge)

    assert _has(_structural_gaps(section, STANDARD), "3 à 5 options distinctes")
    issues = section_issues(section, None, STANDARD)
    assert issues and issues.blocking_count >= 1


def test_mapping_dropping_the_challenge_is_caught_even_when_the_raw_check_is_clean():
    """Défi aux options toutes distinctes... mais dont les textes tronqués à l'API se confondent : seul le contrôle
    rejoué sur le résultat du mapping le voit."""
    long_prefix = "x" * CYCLE_OPTION_MAX

    def mutate(section: dict[str, Any]) -> None:
        section["cycle"]["challenge"].update(
            options=[long_prefix + " un", long_prefix + " deux", long_prefix + " trois"], correct_indices=[0]
        )

    section = _section(mutate)

    assert cycle_structural_gaps(section, STANDARD) == []  # 3 textes distincts avant troncature
    assert _has(cycle_gaps(section, STANDARD, _map_quiz_question), "Défi inexploitable après nettoyage")


@pytest.mark.asyncio
async def test_duplicate_options_course_ends_with_an_incomplete_section_never_an_absent_one():
    client = _fake_gemini([cycle_section(mutate=_duplicate_options_challenge)])

    response = await _generate(client)

    dev = _dev(response)
    assert dev.title == "Sous-réseaux" and dev.incomplete is True and dev.cycle is None
    assert is_incomplete_section_dict(dev.model_dump(mode="json"))


def _drop_two_steps(section: dict[str, Any]) -> None:
    steps = _steps(section)
    steps[0]["support"]["cases"] = [{"label": lbl, "category_index": 0} for lbl in ("x", "X", "x")]
    steps[3]["support"]["order_items"] = ["a", "A", "a"]


def test_steps_dropped_by_the_mapping_below_the_minimum_are_blocking():
    """Même écart sur les étapes : support mappé à None (doublons), 2 étapes restantes pour un minimum de 3."""
    section = _section(_drop_two_steps)

    assert cycle_structural_gaps(section, STANDARD) == []
    gaps = cycle_gaps(section, STANDARD, _map_quiz_question)
    assert _has(gaps, "mini-étape 1", "inexploitable après nettoyage") and _has(gaps, "mini-étape 4")
    assert _has(gaps, "2 mini-étape(s) exploitable(s)", "3 attendues")
    assert section_issues(section, None, STANDARD).blocking


def test_reconstruction_lost_to_the_mapping_is_blocking():
    def mutate(section: dict[str, Any]) -> None:
        steps = _steps(section)
        steps[0]["support"] = {  # 2 catégories sur le papier, une seule une fois les doublons retirés
            "kind": "classify", "categories": ["A", "a"],
            "cases": [{"label": str(i), "category_index": i % 2} for i in range(3)],
        }
        steps[2]["support"] = _table_support("tableau")
        steps[3]["support"] = _table_support("autre tableau")
        steps[3]["support"]["caption"] = "autre"

    section = _section(mutate)

    assert cycle_structural_gaps(section, STANDARD) == []
    assert _has(cycle_gaps(section, STANDARD, _map_quiz_question), "reconstruction exploitable après nettoyage")


def test_reference_section_has_no_mapped_gap():
    assert _gaps() == []


def test_finalize_turns_a_mapping_loss_into_the_incomplete_fallback():
    section = _section(_duplicate_options_challenge)
    planned = [ApiPlannedSection(type="development", title="Sous-réseaux", objective="o", order=2)]

    (final,) = _finalize_incomplete([section], planned, STANDARD)

    assert is_incomplete_section(final)


# ─── 2. Repli RG-15 : la reconstruction est exigée toujours ─────────────────────────────────────────


def _no_reconstruction(section: dict[str, Any]) -> None:
    steps = _steps(section)
    steps[0]["support"] = _table_support("t1")
    steps[2]["support"] = {"kind": "chart", "caption": "c", "chart": {"kind": "bar", "labels": ["a"], "series": [{"name": "s", "values": [1.0]}], "steps": [{"caption": "x"}, {"caption": "y"}]}}
    steps[3]["support"] = {"kind": "diagram", "caption": "d", "diagram": {"kind": "flowchart", "mermaid": "flowchart TD\nA[a] --> B[b]"}}


@pytest.mark.parametrize("brief", [None, {"step_number": 1, "brief": "Un curseur."}])
def test_reconstruction_is_required_with_or_without_a_widget_brief(brief):
    def mutate(section: dict[str, Any]) -> None:
        _no_reconstruction(section)
        section["widget_brief"] = brief

    assert _has(cycle_structural_gaps(_section(mutate), STANDARD), "aucune mini-étape de reconstruction")


@pytest.mark.parametrize("step_number", [9, 0, -1, None])
def test_widget_brief_step_number_is_a_blocking_check(step_number):
    """Sonde de la review : step_number=9 sans reconstruction manquante donnait 0 écart."""
    brief = {"brief": "Un curseur."} if step_number is None else {"step_number": step_number, "brief": "Un curseur."}

    gaps = cycle_structural_gaps(_section(lambda s: s.update(widget_brief=brief)), STANDARD)

    assert _has(gaps, "widget_brief", "step_number", "1 à 4")


def test_widget_brief_without_any_field_does_not_raise():
    section = _section(lambda s: s.update(widget_brief={}))

    assert section.widget_brief is not None and section.widget_brief.step_number is None
    assert _has(cycle_structural_gaps(section, STANDARD), "widget_brief")


# ─── 4. RG-14 : dépassement de maximum = soft + troncature ────────────────────────────────────────────


def _classify_cases(count: int) -> Mutate:
    def mutate(section: dict[str, Any]) -> None:
        _steps(section)[0]["support"]["cases"] = [{"label": f"cas {i}", "category_index": i % 2} for i in range(count)]

    return mutate


def _long_trap_entry(section: dict[str, Any]) -> None:
    entry = section["cycle"]["traps"]["entries"][0]
    entry.update(statements=[f"Affirmation {i}" for i in range(5)], wrong_index=4)


def _four_questions(section: dict[str, Any]) -> None:
    question = section["cycle"]["trial"]["questions"][0]
    section["cycle"]["trial"]["questions"] = [dict(question, question=f"Question {i} ?") for i in range(3)]


def _wide_challenge(section: dict[str, Any]) -> None:
    section["cycle"]["challenge"].update(
        options=["o1", "o2", "o3", "o4", "o5", "bonne"], correct_indices=[5, 5],
    )


def _order_items(count: int) -> Mutate:
    def mutate(section: dict[str, Any]) -> None:
        _steps(section)[3]["support"]["order_items"] = [f"geste {i}" for i in range(count)]

    return mutate


def _five_blanks(section: dict[str, Any]) -> None:
    support = _steps(section)[2]["support"]
    support["table"]["rows"] = [[f"r{i}", str(i), str(i * 2)] for i in range(5)]
    support["blanks"] = [{"row": i, "col": 1} for i in range(5)]


def _five_prediction_options(section: dict[str, Any]) -> None:
    prediction = _steps(section)[1]["prediction"]
    prediction.update(options=["32", "64", "128", "256", "512"], correct_index=1)


@pytest.mark.parametrize(("mutate", "needle"), [
    (_classify_cases(7), "cas à classer"),
    (_order_items(7), "éléments à remettre dans l'ordre"),
    (_five_blanks, "cases masquées"),
    (_five_prediction_options, "prédiction avec plus de 4 options"),
    (_long_trap_entry, "plus de 4 affirmations"),
    (_four_questions, "plus de 2 questions"),
    (_wide_challenge, "Défi : plus de 5 options"),
])
def test_a_maximum_overrun_is_soft_never_blocking(mutate, needle):
    section = _section(mutate)

    assert _has(cycle_bounds_issues(section), needle)
    assert _has(cycle_form_issues(section, STANDARD), needle)
    assert cycle_structural_gaps(section, STANDARD) == []  # jamais de réparation pour un excès
    assert not section_issues(section, None, STANDARD)  # ni d'appel Gemini
    assert _has(list(section_issues(section, None, STANDARD).form), needle)


def test_duplicate_correct_indices_are_soft_and_deduplicated_by_the_mapping():
    section = _section(_wide_challenge)

    assert _has(cycle_bounds_issues(section), "en doublon")
    for seed in range(4):
        challenge = _mapped(_wide_challenge, seed)["challenge"]
        assert [challenge["options"][i] for i in challenge["correct_indices"]] == ["bonne"]


def test_mapping_truncates_to_the_contract_maxima_keeping_the_right_answers():
    cycle = _mapped(lambda s: (_classify_cases(7)(s), _order_items(7)(s), _long_trap_entry(s), _four_questions(s), _wide_challenge(s)))

    assert len(cycle["steps"][0]["support"]["cases"]) == 6
    assert cycle["steps"][3]["support"]["order_items"] == [f"geste {i}" for i in range(6)]
    assert len(cycle["challenge"]["options"]) == 5 and len(cycle["challenge"]["correct_indices"]) == 1
    assert len(cycle["trial"]["questions"]) == 2
    entry = cycle["traps"]["entries"][0]
    assert len(entry["statements"]) == 4 and entry["statements"][entry["wrong_index"]] == "Affirmation 4"


def test_mapping_keeps_the_blank_and_prediction_maxima():
    cycle = _mapped(lambda s: (_five_blanks(s), _five_prediction_options(s)))

    assert len(cycle["steps"][2]["support"]["blanks"]) == 4
    prediction = cycle["steps"][1]["prediction"]
    assert len(prediction["options"]) == 4 and prediction["options"][prediction["correct_index"]] == "64"


def test_order_criterion_is_truncated_to_the_contract_80_characters():
    cycle = _mapped(lambda s: _steps(s)[3]["support"].update(order_criterion="mot " * 40))

    assert len(cycle["steps"][3]["support"]["order_criterion"]) <= 80


# ─── 5. Champs omis et `kind` inconnu : un écart, jamais une exception ───────────────────────────────


def _breakers() -> list[tuple[str, Mutate, str]]:
    return [
        ("support omis", lambda s: _steps(s)[1].pop("support"), "mini-étape 2 : support sans `kind` valide"),
        ("kind omis", lambda s: _steps(s)[1]["support"].pop("kind"), "mini-étape 2 : support sans `kind` valide"),
        ("kind inconnu", lambda s: _steps(s)[1]["support"].update(kind="slider"), "mini-étape 2 : support sans `kind` valide"),
        ("kind non texte", lambda s: _steps(s)[1]["support"].update(kind=["chart"]), "mini-étape 2 : support sans `kind` valide"),
        ("kind de prédiction omis", lambda s: _steps(s)[1]["prediction"].pop("kind"), "prédiction sans `kind` valide"),
        ("kind de prédiction inconnu", lambda s: _steps(s)[1]["prediction"].update(kind="slider"), "prédiction sans `kind` valide"),
        ("case masquée sans coordonnées", lambda s: _steps(s)[2]["support"].update(blanks=[{}]), "case masquée hors du tableau"),
        ("cas sans catégorie", lambda s: _steps(s)[0]["support"].update(cases=[{"label": "a"}] * 3), "catégorie d'un cas hors des catégories"),
        ("épreuve sans question valide", lambda s: s["cycle"]["trial"].update(questions=[{"question": "q", "choices": ["seule"], "correct_indices": [0], "difficulty": "normale", "explanation": "e", "requires_calculation": False}]), "Épreuve `qcm`"),
        ("épreuve sans champ obligatoire", lambda s: s["cycle"]["trial"].update(questions=[{"question": "q"}]), "Épreuve `qcm`"),
    ]


@pytest.mark.parametrize(("label", "mutate", "needle"), _breakers(), ids=[b[0] for b in _breakers()])
def test_an_omitted_field_or_unknown_kind_is_a_blocking_gap_not_an_exception(label, mutate, needle):
    section = _section(mutate)  # ne lève pas

    assert _has(_structural_gaps(section, STANDARD), needle), label
    assert section_issues(section, None, STANDARD).blocking


def test_unknown_trap_mode_and_trial_kind_fall_back_to_the_defaults():
    section = _section(lambda s: (s["cycle"]["traps"].update(mode="zigzag"), s["cycle"]["trial"].update(kind="oral")))

    assert section.cycle.traps.mode.value == "spot" and section.cycle.trial.kind.value == "qcm"


def test_mapping_ignores_a_step_without_kind_and_never_raises():
    cycle = _mapped(lambda s: _steps(s)[1]["support"].update(kind="slider"))

    assert [step["support"]["kind"] for step in cycle["steps"]] == ["classify", "fill_table", "order"]


def test_step_models_accept_missing_fields():
    assert StepSupport().kind is None and StepSupport.model_validate({"kind": "?"}).kind is None
    assert StepPrediction.model_validate({"kind": 3}).kind is None
    assert StepSupport.model_validate({"kind": "match"}).kind is SupportKind.MATCH


def test_the_gemini_schema_still_requires_the_tolerant_fields_and_forbids_null():
    defs = CycleSectionsBatchSchema.model_json_schema()["$defs"]

    for name, field in [("StepSupport", "kind"), ("StepPrediction", "kind"), ("WidgetBrief", "step_number"),
                        ("CellRef", "row"), ("CellRef", "col"), ("ClassifyItem", "category_index")]:
        assert field in defs[name]["required"], (name, field)
        assert "anyOf" not in defs[name]["properties"][field], (name, field)
    assert "support" in defs["CycleStep"]["required"]


@pytest.mark.asyncio
async def test_a_broken_section_does_not_cost_the_whole_batch():
    """Un `kind` inconnu dans UNE section ne fait plus lever ValidationError sur tout le lot : la section est
    réparée, l'autre section du lot reste intacte."""
    broken = cycle_section("A", lambda s: _steps(s)[1]["support"].update(kind="slider"))
    client = _fake_gemini([broken, cycle_section("B")], [cycle_section("A")])

    response = await _generate(client, titles=("A", "B"), batch_size=2)

    sections = [s for s in response.sections if s.title in ("A", "B")]
    assert [s.title for s in sections] == ["A", "B"]
    assert all(s.cycle is not None and not s.incomplete for s in sections)
    prompts = [c.kwargs["raw_answer"] for c in client.format_structured.call_args_list]
    assert any("support sans `kind` valide" in p for p in prompts)


# ─── 6. Épreuve bornée (mapping et API) ───────────────────────────────────────────────────────────────


def _oversized_question(section: dict[str, Any]) -> None:
    question = section["cycle"]["trial"]["questions"][0]
    question.update(
        question="Question " * 200,
        choices=[f"Choix {i} " + "mot " * 80 for i in range(8)],
        correct_indices=[7],
        explanation="Explication " * 100,
        explanation_per_choice=["Retour " * 100 for _ in range(8)],
    )


def test_trial_question_is_truncated_before_the_api_sees_it():
    for seed in range(4):
        (question,) = _mapped(_oversized_question, seed)["trial"]["questions"]
        assert len(question["question"]) <= CYCLE_QUESTION_MAX and len(question["explanation"]) <= 400
        assert len(question["options"]) == 5 and all(len(o) <= CYCLE_OPTION_MAX for o in question["options"])
        assert len(question["explanation_per_choice"]) == 5
        assert all(len(e) <= 400 for e in question["explanation_per_choice"])
        right = question["options"][question["correct_option_indices"][0]]
        assert right.startswith("Choix 7")  # la bonne réponse, d'abord à la fin, survit à la troncature


def test_trial_question_with_duplicate_choices_keeps_per_choice_feedback_aligned():
    def mutate(section: dict[str, Any]) -> None:
        question = section["cycle"]["trial"]["questions"][0]
        question.update(
            choices=["32", "32", "64", "128"], correct_indices=[2],
            explanation_per_choice=["faux 1", "faux 1 bis", "Exact.", "Trop."],
        )

    for seed in range(4):
        (question,) = _mapped(mutate, seed)["trial"]["questions"]
        assert len(question["options"]) == 3
        right = question["correct_option_indices"][0]
        assert question["options"][right] == "64" and question["explanation_per_choice"][right] == "Exact."


def test_api_trial_rejects_unbounded_questions():
    base = _mapped()["trial"]
    too_long = dict(base, questions=[dict(base["questions"][0], question="q" * (CYCLE_QUESTION_MAX + 1))])
    too_many = dict(base, questions=[dict(base["questions"][0], options=list("abcdef"), correct_option_indices=[0])])
    long_feedback = dict(base, questions=[dict(base["questions"][0], explanation_per_choice=["x" * 401] * 3)])
    matching = dict(base, questions=[dict(base["questions"][0], kind="matching")])

    assert ApiCycleTrial.model_validate(base).questions
    for bad in (too_long, too_many, long_feedback, matching):
        with pytest.raises(ValidationError):
            ApiCycleTrial.model_validate(bad)


# ─── 7. Tolérance de prédiction ───────────────────────────────────────────────────────────────────────


def _value_prediction(value: float, tolerance: float | None) -> Mutate:
    def mutate(section: dict[str, Any]) -> None:
        _steps(section)[1]["prediction"] = {
            "kind": "value", "question": "Combien ?", "expected_value": value, "tolerance": tolerance, "outcome": "En réalité.",
        }

    return mutate


@pytest.mark.parametrize("value", [64.0, 1000.0, -4.0, 0.0])
def test_integer_prediction_gets_a_zero_tolerance(value):
    prediction = _mapped(_value_prediction(value, 5.0))["steps"][1]["prediction"]

    assert prediction["tolerance"] == 0


def test_non_integer_prediction_keeps_its_tolerance_up_to_ten_percent():
    assert _mapped(_value_prediction(2.5, 0.2))["steps"][1]["prediction"]["tolerance"] == pytest.approx(0.2)
    assert _mapped(_value_prediction(2.5, 1.0))["steps"][1]["prediction"]["tolerance"] == pytest.approx(0.25)


def test_tolerance_above_ten_percent_and_non_zero_integer_tolerance_are_soft():
    over = _section(_value_prediction(2.5, 1.0))
    integer = _section(_value_prediction(64.0, 3.0))
    fine = _section(_value_prediction(2.5, 0.1))

    assert _has(cycle_bounds_issues(over), "tolérance supérieure à 10 %")
    assert _has(cycle_bounds_issues(integer), "tolérance non nulle pour un entier")
    assert not any("tolérance" in issue for issue in cycle_bounds_issues(fine))
    assert cycle_structural_gaps(over, STANDARD) == [] and not section_issues(integer, None, STANDARD)


# ─── 8. API : le défi est contrôlé comme les prédictions ─────────────────────────────────────────────


@pytest.mark.parametrize("indices", [[4], [-1], [0, 0], [0, 1, 2], []])
def test_api_challenge_checks_correct_indices(indices):
    data = {"question": "q", "options": ["a", "b", "c"], "correct_indices": indices}

    with pytest.raises(ValidationError):
        ApiCycleChallenge.model_validate(data)


def test_api_challenge_accepts_valid_indices():
    assert ApiCycleChallenge.model_validate({"question": "q", "options": ["a", "b", "c"], "correct_indices": [0, 2]})


# ─── 9. Instructions ─────────────────────────────────────────────────────────────────────────────────


def test_instructions_exempt_the_challenge_from_the_multi_answer_wording_and_require_reconstruction():
    from pathlib import Path

    text = (Path(__file__).resolve().parent.parent / "instruction" / "course_generation_instructions.md").read_text(encoding="utf-8")

    assert "does NOT apply to the Défi of the cycle format" in text
    assert "ALWAYS use a reconstruction kind" in text
    assert "if `widget_brief` is null, at least one step" not in text


# ─── 10. Ajout de section : validation de structure et réparation bornée ─────────────────────────────


def _too_few_steps(section: dict[str, Any]) -> None:
    section["cycle"]["steps"] = section["cycle"]["steps"][:2]


@pytest.mark.asyncio
async def test_added_cycle_section_is_repaired_once_when_invalid():
    gemini = _gemini([cycle_section("Le masque", _too_few_steps)], [cycle_section("Le masque")])
    settings = Settings(gemini_api_key="k")

    sections, _ = await add_course_sections(_row("standard"), "Parle du masque", gemini, _store(), settings)

    assert gemini.format_structured.await_count == 1 + settings.section_regenerate_max_repairs
    assert not is_incomplete_section(sections[0]) and len(sections[0].cycle.steps) == 4
    assert "2 mini-étape(s) pour 3 à 5" in gemini.format_structured.await_args_list[1].kwargs["raw_answer"]


@pytest.mark.asyncio
async def test_added_cycle_section_still_invalid_becomes_an_incomplete_fallback():
    invalid = cycle_section("Le masque", _duplicate_options_challenge)
    gemini = _gemini([invalid], [invalid], [invalid], [invalid])
    settings = Settings(gemini_api_key="k")

    sections, _ = await add_course_sections(_row("standard"), "Parle du masque", gemini, _store(), settings)

    assert gemini.format_structured.await_count <= 1 + max(1, settings.section_regenerate_max_repairs)
    assert [s.title for s in sections] == ["Le masque"] and is_incomplete_section(sections[0])
    mapped = _map_sections_to_course_sections(sections, start_index=3, depth="standard")
    assert len(mapped) == 1 and mapped[0].incomplete is True and mapped[0].id == "3"


@pytest.mark.asyncio
async def test_valid_added_cycle_section_costs_no_extra_call():
    gemini = _gemini([cycle_section("Le masque")])

    await add_course_sections(_row("standard"), "Parle du masque", gemini, _store(), Settings(gemini_api_key="k"))

    assert gemini.format_structured.await_count == 1


def test_a_section_without_cycle_in_a_cycle_batch_gets_cycle_wording_in_its_repair_message():
    old = Section.model_validate({"type": "development", "title": "S", "blocks": [], "subsections": []})

    blocking = section_issues(old, None, STANDARD).blocking

    assert len(blocking) == 1 and "objet `cycle` absent" in blocking[0] and "sous-section" not in blocking[0]


# ─── Nits ────────────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("question", [
    "Choisissez les deux bonnes réponses parmi ces options.",
    "Une seule réponse est correcte. Laquelle ?",
    "Cochez toutes les bonnes réponses.",
    "Il y a plusieurs réponses possibles.",
])
def test_challenge_announcing_the_count_is_flagged(question):
    assert _ANNOUNCES_COUNT_RE.search(question)


@pytest.mark.parametrize("question", [
    "Parmi ces deux protocoles, lesquels sont de bonnes pratiques ?",
    "On compare trois réseaux : lesquels ont les bonnes tailles de masque ?",
    "Quelles affirmations sont vraies ?",
])
def test_challenge_not_announcing_the_count_is_not_flagged(question):
    assert not _ANNOUNCES_COUNT_RE.search(question)


def test_numbers_are_not_merged_across_a_plain_space():
    assert _numbers("3 4") == [3.0, 4.0]
    assert _numbers("de 10 à 20 puis 5") == [10.0, 20.0, 5.0]
    assert _numbers("12 345 adresses") == [12345.0]
    assert _numbers("1 000,5") == [1000.5]
    assert _numbers("1 0000") == [1.0, 0.0]
