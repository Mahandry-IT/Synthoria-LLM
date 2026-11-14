"""Format « cycle » : contrôle de structure bloquant, règles de forme RG-3 / RG-4 (soft), prédiction déjà répondue."""

from collections.abc import Callable
from typing import Any

import pytest

from app.schemas.course_generation import Section
from app.services.course_depth import PROFILES, get_profile, render_cycle_rules
from app.services.cycle_validation import cycle_structural_gaps, prediction_reveal_issues, reveals_answer
from app.services.visual_validation import cycle_form_issues, visual_issues
from tests.cycle_fixtures import cycle_section, old_format_section

STANDARD = get_profile("standard")


def _section(mutate: Callable[[dict[str, Any]], None] | None = None) -> Section:
    return Section.model_validate(cycle_section(mutate=mutate))


def _gaps(mutate: Callable[[dict[str, Any]], None], profile=STANDARD) -> list[str]:
    return cycle_structural_gaps(_section(mutate), profile)


def _form(mutate: Callable[[dict[str, Any]], None], profile=STANDARD) -> list[str]:
    return cycle_form_issues(_section(mutate), profile)


def _steps(section: dict[str, Any]) -> list[dict[str, Any]]:
    return section["cycle"]["steps"]


def _has(issues: list[str], *words: str) -> bool:
    return any(all(w in issue for w in words) for issue in issues)


# ─── La section de référence est conforme ───────────────────────────────────────────────


def test_reference_cycle_section_has_no_gap_and_no_form_issue():
    section = _section()

    assert cycle_structural_gaps(section, STANDARD) == []
    assert cycle_form_issues(section, STANDARD) == []
    assert prediction_reveal_issues(section) == []


def test_section_without_cycle_is_not_concerned():
    old = Section.model_validate(old_format_section())

    assert cycle_structural_gaps(old, STANDARD) == [] and cycle_form_issues(old, STANDARD) == []
    assert prediction_reveal_issues(old) == []


# ─── Structure : problèmes BLOQUANTS ─────────────────────────────────────────────────────


@pytest.mark.parametrize(("label", "mutate", "needle"), [
    ("question du défi vide", lambda s: s["cycle"]["challenge"].update(question=" "), "Défi : question vide"),
    ("2 options seulement", lambda s: s["cycle"]["challenge"].update(options=["a", "b"], correct_indices=[0]), "3 à 5 options"),
    ("aucune bonne réponse", lambda s: s["cycle"]["challenge"].update(correct_indices=[]), "correct_indices"),
    ("index hors des options", lambda s: s["cycle"]["challenge"].update(correct_indices=[1, 9]), "correct_indices"),
    ("toutes les options correctes", lambda s: s["cycle"]["challenge"].update(correct_indices=[0, 1, 2, 3]), "jamais désigner toutes"),
    ("options en double (moins de 3 distinctes)", lambda s: s["cycle"]["challenge"].update(options=["a", "A", "a"], correct_indices=[0]), "3 à 5 options distinctes"),
])
def test_challenge_gaps(label, mutate, needle):
    assert _has(_gaps(mutate), needle), label


def test_steps_count_outside_the_mode_range_is_blocking():
    def too_few(section):
        del _steps(section)[2:]

    def too_many(section):
        base = _steps(section)
        kinds = [{"kind": "order", "order_items": ["a", "b", "c"], "order_criterion": "c"},
                 {"kind": "classify", "categories": ["x", "y"], "cases": [{"label": "1", "category_index": 0}, {"label": "2", "category_index": 1}, {"label": "3", "category_index": 0}]}]
        for i in range(2):
            base.append({"text": "Une étape.", "action": "Agis.", "support": kinds[i % 2], "prediction": None})

    assert _has(_gaps(too_few), "2 mini-étape(s) pour 3 à 5")
    assert _has(_gaps(too_many), "6 mini-étape(s) pour 3 à 5")


def test_steps_range_follows_the_course_mode_and_is_wide_without_profile():
    def three_steps(section):
        del _steps(section)[3:]

    assert _gaps(three_steps) == []  # 3 est accepté en standard (3 à 5)
    assert _has(_gaps(three_steps, get_profile("approfondi")), "pour 5 à 7")  # profil non activé, mais défini
    assert _gaps(three_steps, None) == []  # sans profil : plage de tous les modes


def test_every_mode_has_a_cycle_profile_but_only_standard_uses_it():
    assert {name: p.use_cycle for name, p in PROFILES.items()} == {"express": False, "standard": True, "approfondi": False}
    assert all(p.cycle is not None for p in PROFILES.values())


def test_empty_step_text_is_blocking():
    assert _has(_gaps(lambda s: _steps(s)[1].update(text="")), "mini-étape 2 : texte vide")


@pytest.mark.parametrize(("label", "mutate", "needle"), [
    ("tableau sans ligne", lambda s: _steps(s)[2]["support"].update(table={"headers": ["a", "b", "c"], "rows": []}), "tableau vide"),
    ("tableau absent", lambda s: _steps(s)[2]["support"].update(table=None), "tableau vide"),
    ("tableau de cellules vides", lambda s: _steps(s)[2]["support"].update(table={"headers": ["a", "b"], "rows": [["", " "]]}), "tableau vide"),
    ("fill_table sans case masquée", lambda s: _steps(s)[2]["support"].update(blanks=[]), "cases masquées"),
    ("case masquée hors tableau", lambda s: _steps(s)[2]["support"].update(blanks=[{"row": 5, "col": 0}]), "hors du tableau"),
    ("graphique vide", lambda s: _steps(s)[1]["support"].update(chart={"kind": "bar", "labels": [], "series": []}), "graphique vide"),
    ("schéma vide", lambda s: _steps(s)[1]["support"].update(kind="diagram", chart=None, diagram={"kind": "flowchart", "mermaid": ""}), "schéma vide"),
    ("classement à 2 cas", lambda s: _steps(s)[0]["support"].update(cases=_steps(s)[0]["support"]["cases"][:2]), "3 à 6 cas"),
    ("classement à 1 catégorie", lambda s: _steps(s)[0]["support"].update(categories=["seule"]), "2 à 4 catégories"),
    ("catégorie d'un cas inexistante", lambda s: _steps(s)[0]["support"]["cases"][0].update(category_index=4), "catégorie d'un cas"),
    ("ordre à 2 éléments", lambda s: _steps(s)[3]["support"].update(order_items=["a", "b"]), "3 à 6 éléments"),
    ("association à 2 paires", lambda s: _steps(s)[3]["support"].update(kind="match", match_pairs=[{"left": "a", "right": "x"}, {"left": "b", "right": "y"}]), "3 à 5 paires"),
    ("association à `right` répétés", lambda s: _steps(s)[3]["support"].update(kind="match", match_pairs=[{"left": "a", "right": "x"}, {"left": "b", "right": "X"}, {"left": "c", "right": "y"}]), "`right`"),
])
def test_support_gaps(label, mutate, needle):
    assert _has(_gaps(mutate), "mini-étape", needle), label


def test_a_support_filled_with_the_wrong_field_group_is_blocking():
    """Défaut observé à l'essai : un support `table` rempli avec `match_pairs` au lieu de `table`."""
    def mutate(section):
        _steps(section)[2]["support"] = {"kind": "table", "caption": "x", "match_pairs": [{"left": "a", "right": "b"}]}

    assert _has(_gaps(mutate), "tableau vide ou absent")


@pytest.mark.parametrize(("label", "prediction", "needle"), [
    ("choice à 1 option", {"kind": "choice", "question": "q", "options": ["a"], "correct_index": 0, "outcome": "o"}, "2 à 4 options"),
    ("choice sans index", {"kind": "choice", "question": "q", "options": ["a", "b"], "outcome": "o"}, "correct_index"),
    ("choice index hors options", {"kind": "choice", "question": "q", "options": ["a", "b"], "correct_index": 2, "outcome": "o"}, "correct_index"),
    ("value sans valeur", {"kind": "value", "question": "q", "outcome": "o"}, "expected_value"),
    ("value non finie", {"kind": "value", "question": "q", "expected_value": float("inf"), "outcome": "o"}, "expected_value"),
    ("question vide", {"kind": "choice", "question": "", "options": ["a", "b"], "correct_index": 0, "outcome": "o"}, "sans `question` ou sans `outcome`"),
    ("outcome vide", {"kind": "choice", "question": "q", "options": ["a", "b"], "correct_index": 0, "outcome": ""}, "sans `question` ou sans `outcome`"),
])
def test_prediction_gaps(label, prediction, needle):
    assert _has(_gaps(lambda s: _steps(s)[1].update(prediction=prediction)), "mini-étape 2", needle), label


def test_valid_prediction_kinds_are_accepted():
    value = {"kind": "value", "question": "Combien de bits ?", "expected_value": 62.0, "tolerance": 0.0, "unit": "hôtes", "outcome": "En réalité 62."}

    assert _gaps(lambda s: _steps(s)[1].update(prediction=value)) == []


def test_vocabulary_traps_and_trial_gaps():
    assert _has(_gaps(lambda s: s["cycle"].update(vocabulary=[])), "Le mot juste")
    assert _has(_gaps(lambda s: s["cycle"].update(vocabulary=[{"term": "", "plain": "x"}])), "Le mot juste")
    assert _has(_gaps(lambda s: s["cycle"]["traps"].update(entries=[])), "Pièges classiques : aucune entrée")
    assert _has(_gaps(lambda s: s["cycle"]["traps"]["entries"][0].update(wrong_index=3)), "Pièges classiques", "index invalide")
    assert _has(_gaps(lambda s: s["cycle"]["traps"]["entries"][0].update(statements=["a", "b"])), "Pièges classiques")
    assert _has(_gaps(lambda s: s["cycle"]["traps"]["entries"][0].update(explanation=" ")), "Pièges classiques")
    assert _has(_gaps(lambda s: s["cycle"]["trial"].update(questions=[])), "Épreuve `qcm`")
    assert _has(_gaps(lambda s: s["cycle"]["trial"].update(kind="exercise_card", card=None)), "Épreuve `exercise_card`")
    assert _has(_gaps(lambda s: s["cycle"]["trial"].update(kind="exercise_card", card={"statement": "s", "hint": "", "steps": [], "result": ""})), "Épreuve `exercise_card`")


def test_spot_rewrite_and_cards_traps_have_their_own_requirements():
    def rewrite_without_options(section):
        section["cycle"]["traps"]["mode"] = "spot_rewrite"

    def cards_without_claim(section):
        section["cycle"]["traps"]["mode"] = "cards"

    assert _has(_gaps(rewrite_without_options), "spot_rewrite")
    assert _has(_gaps(cards_without_claim), "cards")


def test_trial_qcm_with_an_invalid_numeric_question_is_blocking():
    numeric = {"question": "Combien ?", "kind": "numeric", "difficulty": "normale", "explanation": "e", "requires_calculation": True}

    assert _has(_gaps(lambda s: s["cycle"]["trial"].update(questions=[numeric])), "Épreuve `qcm`")


def test_a_section_never_made_of_text_alone_rg_15():
    def no_reconstruction_no_brief(section):
        _steps(section)[0]["support"] = {"kind": "table", "caption": "t", "table": {"headers": ["a", "b", "c"], "rows": [["1", "2", "3"]]}}
        _steps(section)[2]["support"] = {"kind": "chart", "caption": "c", "chart": {"kind": "bar", "labels": ["a"], "series": [{"name": "s", "values": [1.0]}], "steps": [{"caption": "x"}, {"caption": "y"}]}}
        _steps(section)[3]["support"] = {"kind": "diagram", "caption": "d", "diagram": {"kind": "flowchart", "mermaid": "flowchart TD\nA[a] --> B[b]"}}
        section["widget_brief"] = None

    def brief_replaces_reconstruction(section):
        no_reconstruction_no_brief(section)
        section["widget_brief"] = {"step_number": 1, "brief": "Un curseur."}

    # RG-15 : la reconstruction est exigée TOUJOURS, le widget n'étant jamais garanti.
    assert _has(_gaps(no_reconstruction_no_brief), "aucune mini-étape de reconstruction")
    assert _has(_gaps(brief_replaces_reconstruction), "aucune mini-étape de reconstruction")
    assert _gaps(lambda s: s.update(widget_brief=None)) == []  # des étapes de reconstruction suffisent


# ─── Forme : RG-3 et RG-4 (soft) ───────────────────────────────────────────────────────


def test_rg3_two_consecutive_steps_with_the_same_support_kind():
    def mutate(section):
        _steps(section)[3]["support"] = {
            "kind": "fill_table", "caption": "bis", "table": {"headers": ["a", "b", "c"], "rows": [["1", "2", "3"]]}, "blanks": [{"row": 0, "col": 0}],
        }

    assert _has(_form(mutate), "mini-étape 4", "même type de support (fill_table)")


def test_rg3_same_kind_two_steps_apart_is_fine():
    def mutate(section):
        _steps(section)[2]["support"] = _steps(section)[0]["support"]

    assert not _has(_form(mutate), "même type de support")


def test_rg4_text_over_80_words_and_over_the_soft_target():
    long_text = ("mot " * 90).strip() + "."
    mid_text = ("mot " * 60).strip() + "."

    assert _has(_form(lambda s: _steps(s)[0].update(text=long_text)), "mini-étape 1", "90 mots pour un plafond de 80")
    assert _has(_form(lambda s: _steps(s)[0].update(text=mid_text)), "mini-étape 1", "environ 40 visés")
    assert not _has(_form(lambda s: _steps(s)[0].update(text=("mot " * 40).strip() + ".")), "mots")


def test_rg4_more_than_three_sentences():
    assert _has(_form(lambda s: _steps(s)[0].update(text="Un. Deux. Trois. Quatre.")), "plus de 3 phrases")


def test_rg4_table_limits_rows_columns_and_cell_words():
    rows = [[str(r), "x", "y"] for r in range(6)]

    def six_rows(section):
        _steps(section)[2]["support"].update(table={"headers": ["a", "b", "c"], "rows": rows})

    def five_cols(section):
        _steps(section)[2]["support"].update(table={"headers": list("abcde"), "rows": [list("12345")]})

    def two_cols(section):
        _steps(section)[2]["support"].update(table={"headers": ["a", "b"], "rows": [["1", "2"]]})

    def long_cell(section):
        _steps(section)[2]["support"].update(table={"headers": ["a", "b", "c"], "rows": [["un deux trois quatre cinq six sept huit neuf", "1", "2"]]})

    assert _has(_form(six_rows), "6 lignes (plafond 5)")
    assert _has(_form(five_cols), "5 colonnes")
    assert _has(_form(two_cols), "2 colonnes")
    assert _has(_form(long_cell), "plus de 8 mots")


def test_rg4_a_table_of_five_rows_and_four_columns_is_accepted():
    def mutate(section):
        _steps(section)[2]["support"].update(table={"headers": list("abcd"), "rows": [[f"{r}{c}" for c in "abcd"] for r in range(5)]}, blanks=[{"row": 0, "col": 0}])

    assert _form(mutate) == []


def test_rg4_one_formula_per_step_and_no_mix_of_prose_formula_and_table():
    two_formulas = "On pose $a = 2$ puis $b = 3$."
    display = "Le résultat vaut $$x = \\frac{a}{b}$$ et se lit ainsi."
    table_in_text = "Voici les valeurs.\n| a | b |\n| 1 | 2 |"

    assert _has(_form(lambda s: _steps(s)[0].update(text=two_formulas)), "plus d'une formule")
    assert _has(_form(lambda s: _steps(s)[0].update(text=display)), "mélange prose et formule")
    assert _has(_form(lambda s: _steps(s)[0].update(text=table_in_text)), "mélange prose et formule")
    assert _form(lambda s: _steps(s)[0].update(text="On pose $a = 2$.")) == []


def test_form_checks_on_charts_and_diagrams():
    def static_chart(section):
        _steps(section)[1]["support"]["chart"]["steps"] = []

    def paren_label(section):
        _steps(section)[1]["support"] = {"kind": "diagram", "caption": "d", "diagram": {"kind": "flowchart", "mermaid": "flowchart TD\nA[Évaluation f(s1)] --> B[Fin]"}}

    def big(section):
        edges = "\n".join(f"N{i}[n{i}] --> N{i + 1}[n{i + 1}]" for i in range(9))
        _steps(section)[1]["support"] = {"kind": "diagram", "caption": "d", "diagram": {"kind": "flowchart", "mermaid": "flowchart TD\n" + edges}}

    assert _has(_form(static_chart), "graphique sans `steps`")
    assert _has(_form(paren_label), "parenthèses sans guillemets")
    assert _has(_form(big), "plus de 8 nœuds")


def test_form_checks_on_the_challenge():
    def long_feedback(section):
        section["cycle"]["challenge"]["feedback"] = "Première phrase. Deuxième phrase."

    def announces(section):
        section["cycle"]["challenge"]["question"] = "Choisissez les deux bonnes réponses parmi ces options."

    assert _has(_form(long_feedback), "Défi : correction de plus d'une ligne")
    assert _has(_form(announces), "annonce le nombre de bonnes réponses")


def test_form_checks_follow_the_mode_profile():
    def three_predictions(section):
        for i in (0, 1, 3):
            _steps(section)[i]["prediction"] = {"kind": "choice", "question": "q", "options": ["a", "b"], "correct_index": 0, "outcome": "o"}

    def no_prediction(section):
        _steps(section)[1]["prediction"] = None

    def four_traps(section):
        entries = section["cycle"]["traps"]["entries"]
        entries += [dict(entries[0]), dict(entries[1])]

    assert _has(_form(three_predictions), "3 prédiction(s) pour 1 à 2")
    assert _has(_form(no_prediction), "0 prédiction(s)")
    assert _has(_form(lambda s: s["cycle"]["traps"].update(mode="cards")), "mode `cards` au lieu de `spot`")
    assert _has(_form(four_traps), "Pièges : 4 entrée(s)")
    assert _has(_form(lambda s: s["cycle"].update(vocabulary=s["cycle"]["vocabulary"][:2])), "2 terme(s) pour 3 à 6")
    assert _has(_form(lambda s: s.update(key_takeaways=["une seule"])), "1 `key_takeaways` pour 3")


def test_form_checks_flag_consecutive_predictions_while_a_bad_widget_brief_step_is_blocking():
    def consecutive(section):
        _steps(section)[2]["prediction"] = {"kind": "choice", "question": "q", "options": ["a", "b"], "correct_index": 0, "outcome": "o"}

    assert _has(_form(consecutive), "prédictions sur des mini-étapes consécutives")
    assert not _has(_form(lambda s: s["widget_brief"].update(step_number=9)), "widget_brief")
    assert _has(_gaps(lambda s: s["widget_brief"].update(step_number=9)), "widget_brief", "step_number", "1 à 4")


def test_form_checks_flag_over_long_character_fields():
    assert _has(_form(lambda s: s["cycle"].update(thread="x" * 200)), "fil rouge : plus de 140 caractères")
    assert _has(_form(lambda s: s["cycle"]["challenge"]["options"].__setitem__(0, "x" * 100)), "option du défi : plus de 90")
    assert _has(_form(lambda s: _steps(s)[1]["prediction"]["options"].__setitem__(0, "x" * 70)), "option de prédiction : plus de 60")


def test_vocabulary_definition_over_20_words():
    assert _has(_form(lambda s: s["cycle"]["vocabulary"][0].update(plain=("mot " * 25).strip())), "définition de plus de 20 mots")


def test_visual_issues_delegates_to_the_cycle_rules_for_a_cycle_section():
    section = _section(lambda s: _steps(s)[0].update(text=("mot " * 90).strip() + "."))

    assert visual_issues(section, STANDARD) == cycle_form_issues(section, STANDARD)
    assert not _has(visual_issues(section, STANDARD), "bloc visuel")  # jamais les règles de l'ancien format


# ─── Prédiction déjà répondue ────────────────────────────────────────────────────────────


def _choice(options: list[str], correct: int):
    from app.schemas.course_generation import StepPrediction

    return StepPrediction.model_validate({"kind": "choice", "question": "q", "options": options, "correct_index": correct, "outcome": "o"})


def test_reveal_when_the_right_option_is_written_in_the_text():
    prediction = _choice(["32 adresses", "64 adresses", "128 adresses"], 1)

    assert reveals_answer(prediction, "Chaque part compte 64 adresses.") is True
    assert reveals_answer(prediction, "On partage 256 adresses en quatre.") is False


def test_reveal_by_lexical_coverage_of_the_right_option_only():
    prediction = _choice(["Le routeur renvoie le paquet à l'émetteur", "Le routeur détruit le paquet", "Le paquet part vers la sortie"], 0)

    assert reveals_answer(prediction, "Ensuite le routeur renvoie le paquet vers son émetteur.") is True
    assert reveals_answer(prediction, "Le paquet arrive au routeur.") is False  # couvre aussi bien les autres options


def test_reveal_of_a_value_prediction_is_the_expected_number_in_the_text():
    from app.schemas.course_generation import StepPrediction

    prediction = StepPrediction.model_validate({"kind": "value", "question": "q", "expected_value": 62.0, "tolerance": 0.0, "outcome": "o"})

    assert reveals_answer(prediction, "Le sous-réseau compte 62 hôtes.") is True
    assert reveals_answer(prediction, "Le sous-réseau compte 64 adresses.") is False


def test_reveal_issues_name_the_step_and_distinguish_own_text_from_earlier_steps():
    def own_text(section):
        _steps(section)[1]["text"] = "On partage 256 adresses : chaque part compte 64 adresses."

    def earlier(section):
        _steps(section)[0]["text"] = "Un réseau /24 compte 256 adresses, soit 64 adresses par part."

    assert _has(prediction_reveal_issues(_section(own_text)), "mini-étape 2", "texte de l'étape")
    assert _has(prediction_reveal_issues(_section(earlier)), "mini-étape 2", "étapes précédentes")


def test_reveal_check_ignores_small_integers_in_earlier_steps():
    def mutate(section):
        _steps(section)[1]["prediction"] = {"kind": "value", "question": "Combien de parts ?", "expected_value": 4.0, "tolerance": 0.0, "outcome": "En réalité 4."}
        _steps(section)[0]["text"] = "On veut 4 parts égales dans le réseau."

    assert not _has(prediction_reveal_issues(_section(mutate)), "étapes précédentes")


# ─── Règles de mode rendues dans le prompt ──────────────────────────────────────────────


def test_render_cycle_rules_states_the_standard_numbers_from_the_profile():
    rules = render_cycle_rules(STANDARD)

    assert "3 à 5 (vise 4) mini-étapes" in rules
    assert "Prédictions : 1 à 2" in rules and "`spot`" in rules and "3 à 6 termes" in rules and "3 puces" in rules
    assert "`subsections`" in rules  # l'ancien format n'est pas produit pour ces sections


def test_render_cycle_rules_adapts_to_each_mode():
    assert "3 mini-étapes" in render_cycle_rules(get_profile("express")) and "`cards`" in render_cycle_rules(get_profile("express"))
    assert "5 à 7 (vise 5)" in render_cycle_rules(get_profile("approfondi")) and "`spot_rewrite`" in render_cycle_rules(get_profile("approfondi"))
