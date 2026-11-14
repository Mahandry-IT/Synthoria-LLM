"""Mapping du `cycle` d'une section générée par Gemini vers le `cycle` de l'API.

Tout ce que le modèle écrit est non fiable. Ce module :
- tronque les textes aux bornes de l'API (`CYCLE_*_MAX`) et écarte, élément par élément, ce qui est vide ou
  invalide (une mini-étape sans support exploitable, une option vide, une prédiction aux index faux...) : un
  élément écarté n'invalide jamais le cours, et rien ne lève ;
- MÉLANGE les options du défi, des questions de l'épreuve et des prédictions `choice`, ainsi que les
  affirmations et reformulations des pièges, puis recalcule les index des bonnes réponses : la place de la
  bonne réponse ne dépend plus de ce que le modèle a écrit (il met trop souvent l'index 0) ;
- retire la valeur attendue de la carte-exercice et borne la tolérance des prédictions `value` à 10 %.

Le résultat est un dict conforme à `ApiSectionCycle`, ou None si le défi ou toutes les mini-étapes sont
inexploitables (la section n'a alors plus de cycle). Ce cas ne doit JAMAIS arriver en silence : `cycle_gaps`
rejoue ce mapping pendant la validation de structure (bloquant, donc réparation puis repli « incomplète »), et
`course_generator._map_sections_to_course_sections` remplace en dernier recours la section par un repli
« incomplète » plutôt que de la laisser disparaître.
"""

import logging
import math
import random
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ValidationError

from app.api.schemas import (
    CHART_LABELS_MAX,
    CHART_SERIES_MAX,
    CYCLE_ACTION_MAX,
    CYCLE_CAPTION_MAX,
    CYCLE_CARD_LINE_MAX,
    CYCLE_CARD_STATEMENT_MAX,
    CYCLE_CARD_STEPS_MAX,
    CYCLE_CATEGORY_MAX,
    CYCLE_CELL_MAX,
    CYCLE_CLAIM_MAX,
    CYCLE_EXPLANATION_MAX,
    CYCLE_FEEDBACK_MAX,
    CYCLE_FORMULA_MAX,
    CYCLE_ITEM_MAX,
    CYCLE_LABEL_MAX,
    CYCLE_OPTION_MAX,
    CYCLE_OUTCOME_MAX,
    CYCLE_PLAIN_MAX,
    CYCLE_QUESTION_MAX,
    CYCLE_STEP_TEXT_MAX,
    CYCLE_STEPS_MAX,
    CYCLE_TABLE_COLS_MAX,
    CYCLE_TABLE_ROWS_MAX,
    CYCLE_TERM_MAX,
    CYCLE_THREAD_MAX,
    CYCLE_TRAP_ENTRIES_MAX,
    CYCLE_UNIT_MAX,
    CYCLE_VOCABULARY_MAX,
    WIDGET_BRIEF_MAX,
    ApiChart,
    ApiDiagram,
    ApiSectionCycle,
    ApiWidgetBrief,
    CourseSection,
    CourseTable,
)
from app.schemas.course_generation import (
    ChartData,
    CycleChallenge,
    CycleTraps,
    CycleTrial,
    CycleStep,
    ExerciseCard,
    RECONSTRUCTION_KINDS,
    PredictionKind,
    QuizKind,
    QuizQuestion,
    Section,
    StepPrediction,
    StepSupport,
    SupportKind,
    TableData,
    TrapMode,
    TrialKind,
    WidgetBrief,
    quiz_question_issue,
)
from app.services.course_depth import DepthProfile, estimate_minutes
from app.services.cycle_validation import (
    CHALLENGE_OPTIONS,
    CLASSIFY_CASES,
    CLASSIFY_CATEGORIES,
    FILL_TABLE_BLANKS,
    MATCH_PAIRS,
    ORDER_ITEMS,
    PREDICTION_OPTIONS,
    PREDICTION_TOLERANCE_MAX_RATIO,  # noqa: F401 — règle RG-16, ré-exportée (appliquée ici)
    REWRITE_OPTIONS,
    SPOT_STATEMENTS,
    TRIAL_OPTIONS_MAX,
    TRIAL_QUESTIONS,
    cycle_structural_gaps,
)

logger = logging.getLogger(__name__)

# Maxima du contrat (`cycle_validation`) : un excès n'est jamais bloquant, le mapping tronque.
_DISTRACTORS_MAX = 3
# Critère d'une remise dans l'ordre : 80 caractères au contrat (l'API accepte davantage).
_ORDER_CRITERION_MAX = 80

MapQuestion = Callable[[QuizQuestion], dict[str, Any]]
_RECONSTRUCTION_VALUES = frozenset(kind.value for kind in RECONSTRUCTION_KINDS)


def truncate_text(text: str, limit: int) -> str:
    """Tronque à la limite de mot avec « … » ; un texte dans la limite est rendu tel quel."""
    if len(text) <= limit:
        return text
    cut = text[: limit - 1]
    if text[limit - 1].isspace():  # coupe déjà tombée sur une fin de mot
        return cut.rstrip() + "…"
    head, sep, _ = cut.rpartition(" ")
    return (head if sep and head else cut).rstrip() + "…"


def _clean(value: object, limit: int) -> str:
    """Texte du modèle : espaces de bord retirés, longueur bornée ; "" si ce n'est pas un texte."""
    return truncate_text(value.strip(), limit) if isinstance(value, str) else ""


def _valid_index(value: object, size: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value < size


def _single(index: int | None, size: int) -> set[int]:
    """`{index}` s'il désigne une option existante, sinon l'ensemble vide."""
    return {index} if index is not None and _valid_index(index, size) else set()


def _choice_candidates(
    items: list[str], limit: int, cap: int, correct: set[int]
) -> list[tuple[int, str, bool]]:
    """`(index d'origine, texte, correcte ?)` des options non vides et distinctes, au plus `cap`.

    Deux options de même texte n'en font qu'une, correcte si l'une des deux l'est. Au-delà de `cap`, les bonnes
    réponses sont gardées en priorité (tronquer ne doit jamais retirer la bonne réponse), l'ordre d'origine est
    conservé."""
    unique: dict[str, tuple[int, str, bool]] = {}
    for old, raw in enumerate(items):
        text = _clean(raw, limit)
        if not text:
            continue
        first, first_text, was_correct = unique.get(text.casefold(), (old, text, False))
        unique[text.casefold()] = (first, first_text, was_correct or old in correct)
    candidates = list(unique.values())
    if len(candidates) > cap:
        right = [c for c in candidates if c[2]][:cap]
        kept = {c[0] for c in right}
        wrong = [c for c in candidates if c[0] not in kept][: cap - len(right)]
        candidates = sorted([*right, *wrong])
    return candidates


def _clean_choices(
    items: list[str], limit: int, cap: int, correct: set[int]
) -> tuple[list[str], set[int]]:
    """Options non vides et distinctes (au plus `cap`) et nouveaux index de celles qui sont correctes.

    Écarter une option vide ou en double décale les index : on suit donc chaque option d'origine."""
    candidates = _choice_candidates(items, limit, cap, correct)
    return [c[1] for c in candidates], {position for position, c in enumerate(candidates) if c[2]}


def _shuffled(texts: list[str], correct: set[int], rng: random.Random) -> tuple[list[str], list[int]]:
    """Options mélangées et index (triés) des bonnes réponses dans le nouvel ordre."""
    order = list(range(len(texts)))
    rng.shuffle(order)
    return [texts[i] for i in order], sorted(position for position, old in enumerate(order) if old in correct)


def _validated(model: type[BaseModel], data: dict[str, Any]) -> dict[str, Any] | None:
    try:
        return model.model_validate(data).model_dump(mode="json", exclude_none=True)
    except ValidationError:
        return None


# ─── Défi ────────────────────────────────────────────────────────────────────


def _map_challenge(challenge: CycleChallenge, rng: random.Random) -> dict[str, Any] | None:
    question = _clean(challenge.question, CYCLE_QUESTION_MAX)
    valid = {i for i in challenge.correct_indices if _valid_index(i, len(challenge.options))}
    options, correct = _clean_choices(challenge.options, CYCLE_OPTION_MAX, CHALLENGE_OPTIONS[1], valid)
    # Jamais toutes les options correctes : le défi n'aurait plus rien à départager.
    if not question or len(options) < 2 or not correct or len(correct) >= len(options):
        return None
    mixed, correct_indices = _shuffled(options, correct, rng)
    return {
        "question": question,
        "options": mixed,
        "correct_indices": correct_indices,
        "feedback": _clean(challenge.feedback, CYCLE_FEEDBACK_MAX),
    }


# ─── Supports des mini-étapes ──────────────────────────────────────────────────


def _map_chart(chart: ChartData | None) -> dict[str, Any] | None:
    if chart is None or not chart.labels or not chart.series:
        return None
    # Séries tronquées à la longueur des libellés : un décalage rendrait le graphique faux.
    size = min(CHART_LABELS_MAX, len(chart.labels), *(len(s.values) for s in chart.series))
    if size < 1 or any(not math.isfinite(v) for s in chart.series for v in s.values[:size]):
        return None
    data = {
        "kind": chart.kind.value,
        "caption": _clean(chart.caption, CYCLE_CAPTION_MAX),
        "labels": [_clean(label, CYCLE_CELL_MAX) for label in chart.labels[:size]],
        "series": [
            {"name": _clean(s.name, CYCLE_CELL_MAX) or "Série", "values": s.values[:size]}
            for s in chart.series[:CHART_SERIES_MAX]
        ],
        "steps": [s.model_dump(mode="json") for s in chart.steps],  # nettoyées par ApiChart
    }
    return _validated(ApiChart, data)


def _map_table(table: TableData | None) -> dict[str, Any] | None:
    """Tableau borné ; les cases (et donc les index de `blanks`) gardent leur position d'origine."""
    if table is None:
        return None
    headers = [_clean(h, CYCLE_CELL_MAX) for h in table.headers[:CYCLE_TABLE_COLS_MAX]]
    if not headers:
        return None
    rows = [
        [_clean(cell, CYCLE_CELL_MAX) for cell in row[: len(headers)]] + [""] * (len(headers) - len(row))
        for row in table.rows[:CYCLE_TABLE_ROWS_MAX]
    ]
    if not any(cell for row in rows for cell in row):
        return None
    return _validated(CourseTable, {"caption": _clean(table.caption, CYCLE_CAPTION_MAX), "headers": headers, "rows": rows})


def _map_diagram(support: StepSupport) -> dict[str, Any] | None:
    diagram = support.diagram
    if diagram is None or not diagram.mermaid.strip():
        return None
    return _validated(
        ApiDiagram,
        {"kind": diagram.kind.value, "caption": _clean(diagram.caption, CYCLE_CAPTION_MAX), "mermaid": diagram.mermaid.strip()},
    )


def _map_blanks(support: StepSupport, table: dict[str, Any]) -> tuple[list[dict[str, int]], list[str]]:
    """Cases masquées (dans le tableau, non vides, sans doublon) et leurres (distincts des réponses)."""
    blanks: list[dict[str, int]] = []
    answers: set[str] = set()
    for ref in support.blanks:
        in_table = _valid_index(ref.row, len(table["rows"])) and _valid_index(ref.col, len(table["headers"]))
        answer = table["rows"][ref.row][ref.col] if in_table else ""
        if answer and {"row": ref.row, "col": ref.col} not in blanks and len(blanks) < FILL_TABLE_BLANKS[1]:
            blanks.append({"row": ref.row, "col": ref.col})
            answers.add(answer.casefold())
    distractors: list[str] = []
    for raw in support.distractors:
        text = _clean(raw, CYCLE_CELL_MAX)
        if text and text.casefold() not in answers and text not in distractors and len(distractors) < _DISTRACTORS_MAX:
            distractors.append(text)
    return blanks, distractors


def _map_classify(support: StepSupport, rng: random.Random) -> dict[str, Any] | None:
    categories: list[str] = []
    new_index: dict[int, int] = {}
    for old, raw in enumerate(support.categories):
        name = _clean(raw, CYCLE_CATEGORY_MAX)
        if name and name.casefold() not in {c.casefold() for c in categories} and len(categories) < CLASSIFY_CATEGORIES[1]:
            new_index[old] = len(categories)
            categories.append(name)
    cases: list[dict[str, Any]] = []
    for case in support.cases:
        label = _clean(case.label, CYCLE_LABEL_MAX)
        if (
            label and case.category_index in new_index and len(cases) < CLASSIFY_CASES[1]
            and label.casefold() not in {c["label"].casefold() for c in cases}
        ):
            cases.append({"label": label, "category_index": new_index[case.category_index]})
    if len(categories) < 2 or len(cases) < 2:
        return None
    rng.shuffle(cases)  # le modèle range souvent les cas par catégorie : l'ordre ne doit rien révéler
    return {"categories": categories, "cases": cases}


def _map_order(support: StepSupport) -> dict[str, Any] | None:
    items, _ = _clean_choices(support.order_items, CYCLE_ITEM_MAX, ORDER_ITEMS[1], set())
    if len(items) < 2:
        return None
    return {"order_items": items, "order_criterion": _clean(support.order_criterion, _ORDER_CRITERION_MAX)}


def _map_match(support: StepSupport) -> dict[str, Any] | None:
    pairs: list[dict[str, str]] = []
    rights: set[str] = set()
    for pair in support.match_pairs:
        left, right = _clean(pair.left, CYCLE_LABEL_MAX), _clean(pair.right, CYCLE_LABEL_MAX)
        if left and right and right.casefold() not in rights and len(pairs) < MATCH_PAIRS[1]:
            rights.add(right.casefold())
            pairs.append({"left": left, "right": right})
    return {"match_pairs": pairs} if len(pairs) >= 2 else None


def _table_from_pairs(support: StepSupport) -> dict[str, Any] | None:
    """Rattrapage : un support `table` rempli avec `match_pairs` (et sans tableau) devient un tableau à 2 colonnes."""
    rows = [
        [_clean(pair.left, CYCLE_CELL_MAX), _clean(pair.right, CYCLE_CELL_MAX)]
        for pair in support.match_pairs
    ]
    rows = [row for row in rows if row[0] and row[1]][:CYCLE_TABLE_ROWS_MAX]
    if len(rows) < 2:
        return None
    return _validated(
        CourseTable,
        {"caption": _clean(support.caption, CYCLE_CAPTION_MAX), "headers": ["Élément", "Correspondance"], "rows": rows},
    )


def _map_table_support(support: StepSupport) -> tuple[SupportKind, dict[str, Any] | None]:
    table = _map_table(support.table)
    if table is None and support.kind is SupportKind.TABLE:
        table = _table_from_pairs(support)
    if table is None:
        return support.kind, None
    if support.kind is SupportKind.FILL_TABLE:
        blanks, distractors = _map_blanks(support, table)
        if blanks:
            return support.kind, {"table": table, "blanks": blanks, "distractors": distractors}
        return SupportKind.TABLE, {"table": table}  # rien à compléter : le tableau reste un support valide
    return support.kind, {"table": table}


def _map_support(support: StepSupport, rng: random.Random) -> dict[str, Any] | None:
    """Support d'une mini-étape, ou None s'il est inexploitable (la mini-étape est alors écartée)."""
    kind = support.kind
    payload: dict[str, Any] | None
    if kind is None:
        return None
    if kind is SupportKind.CHART:
        chart = _map_chart(support.chart)
        payload = {"chart": chart} if chart else None
    elif kind is SupportKind.DIAGRAM:
        diagram = _map_diagram(support)
        payload = {"diagram": diagram} if diagram else None
    elif kind in (SupportKind.TABLE, SupportKind.FILL_TABLE):
        kind, payload = _map_table_support(support)
    elif kind is SupportKind.CLASSIFY:
        payload = _map_classify(support, rng)
    elif kind is SupportKind.ORDER:
        payload = _map_order(support)
    else:
        payload = _map_match(support)
    if payload is None:
        return None
    return {"kind": kind.value, "caption": _clean(support.caption, CYCLE_CAPTION_MAX), **payload}


# ─── Prédictions ───────────────────────────────────────────────────────────────


def _map_prediction(prediction: StepPrediction | None, rng: random.Random) -> dict[str, Any] | None:
    """Prédiction (facultative) ; invalide, elle est simplement écartée et la mini-étape garde son support."""
    if prediction is None or prediction.kind is None:
        return None
    question = _clean(prediction.question, CYCLE_QUESTION_MAX)
    outcome = _clean(prediction.outcome, CYCLE_OUTCOME_MAX)
    if not question or not outcome:
        return None
    if prediction.kind is PredictionKind.CHOICE:
        options, new_correct = _clean_choices(
            prediction.options, CYCLE_OPTION_MAX, PREDICTION_OPTIONS[1],
            _single(prediction.correct_index, len(prediction.options)),
        )
        if len(options) < 2 or len(new_correct) != 1:
            return None
        mixed, indices = _shuffled(options, new_correct, rng)
        return {"kind": "choice", "question": question, "options": mixed, "correct_index": indices[0], "outcome": outcome}
    value = prediction.expected_value
    if value is None or not math.isfinite(value):
        return None
    tolerance = prediction.tolerance
    tolerance = 0.0 if tolerance is None or not math.isfinite(tolerance) or tolerance < 0 else tolerance
    if float(value).is_integer():
        tolerance = 0.0  # un entier ou un dénombrement est exact : aucune marge (RG-16)
    mapped: dict[str, Any] = {
        "kind": "value",
        "question": question,
        "expected_value": value,
        "tolerance": min(tolerance, abs(value) * PREDICTION_TOLERANCE_MAX_RATIO),
        "outcome": outcome,
    }
    if unit := _clean(prediction.unit, CYCLE_UNIT_MAX):
        mapped["unit"] = unit
    return mapped


# ─── Pièges ────────────────────────────────────────────────────────────────────


def _map_spot(
    statements: list[str], wrong: int | None, rng: random.Random, cap: int
) -> tuple[list[str], int] | None:
    texts, correct = _clean_choices(statements, CYCLE_CLAIM_MAX, cap, _single(wrong, len(statements)))
    if len(texts) < 2 or len(correct) != 1:
        return None
    mixed, indices = _shuffled(texts, correct, rng)
    return mixed, indices[0]


def _map_trap_entry(entry: Any, mode: TrapMode, rng: random.Random) -> dict[str, Any] | None:
    explanation = _clean(entry.explanation, CYCLE_EXPLANATION_MAX)
    if not explanation:
        return None
    if mode is TrapMode.CARDS:
        claim = _clean(entry.claim, CYCLE_CLAIM_MAX)
        if not claim and _single(entry.wrong_index, len(entry.statements)):
            claim = _clean(entry.statements[entry.wrong_index], CYCLE_CLAIM_MAX)
        return {"claim": claim, "explanation": explanation} if claim else None
    spot = _map_spot(entry.statements, entry.wrong_index, rng, SPOT_STATEMENTS[1])
    if spot is None:
        return None
    mapped: dict[str, Any] = {"statements": spot[0], "wrong_index": spot[1], "explanation": explanation}
    if mode is TrapMode.SPOT_REWRITE:
        rewrite = _map_spot(entry.rewrite_options, entry.correct_rewrite_index, rng, REWRITE_OPTIONS[1])
        if rewrite is None:
            return None
        mapped.update(rewrite_options=rewrite[0], correct_rewrite_index=rewrite[1])
    return mapped


def _map_traps(traps: CycleTraps, rng: random.Random) -> dict[str, Any] | None:
    entries = [m for e in traps.entries if (m := _map_trap_entry(e, traps.mode, rng)) is not None]
    return {"mode": traps.mode.value, "entries": entries[:CYCLE_TRAP_ENTRIES_MAX]} if entries else None


# ─── Épreuve ───────────────────────────────────────────────────────────────────


def _map_trial_question(q: QuizQuestion, rng: random.Random, map_question: MapQuestion) -> dict[str, Any] | None:
    """Question de l'épreuve : choix ou valeur numérique (jamais d'association). Tout texte est borné ; les choix
    (au plus `TRIAL_OPTIONS_MAX`, bonnes réponses gardées) sont mélangés avec leurs retours par option, et les
    index des bonnes réponses recalculés."""
    if q.kind is QuizKind.MATCHING or quiz_question_issue(q) or not q.question.strip():
        return None
    update: dict[str, Any] = {
        "question": _clean(q.question, CYCLE_QUESTION_MAX),
        "explanation": _clean(q.explanation, CYCLE_EXPLANATION_MAX),
        "unit": _clean(q.unit, CYCLE_UNIT_MAX) or None,
    }
    if q.kind is QuizKind.CHOICE:
        picked = _choice_candidates(q.choices, CYCLE_OPTION_MAX, TRIAL_OPTIONS_MAX, set(q.correct_indices))
        if len(picked) < 2 or not any(correct for _, _, correct in picked):
            return None
        order = list(range(len(picked)))
        rng.shuffle(order)
        per_choice = q.explanation_per_choice
        update.update({
            "choices": [picked[i][1] for i in order],
            "correct_indices": sorted(position for position, i in enumerate(order) if picked[i][2]),
            "explanation_per_choice": (
                [_clean(per_choice[picked[i][0]], CYCLE_EXPLANATION_MAX) for i in order]
                if len(per_choice) == len(q.choices) else []
            ),
        })
    return map_question(q.model_copy(update=update))


def _map_card(card: ExerciseCard | None) -> dict[str, Any] | None:
    if card is None:
        return None
    steps = [s for s in (_clean(raw, CYCLE_CARD_LINE_MAX) for raw in card.steps) if s][:CYCLE_CARD_STEPS_MAX]
    statement = _clean(card.statement, CYCLE_CARD_STATEMENT_MAX)
    if not statement or not steps:
        return None
    # Aucune valeur attendue : `expected_value` n'existe plus sur la carte-exercice.
    return {
        "statement": statement,
        "hint": _clean(card.hint, CYCLE_CARD_LINE_MAX),
        "steps": steps,
        "result": _clean(card.result, CYCLE_CARD_LINE_MAX),
    }


def _map_trial(trial: CycleTrial, rng: random.Random, map_question: MapQuestion) -> dict[str, Any] | None:
    def qcm() -> dict[str, Any] | None:
        questions = [m for q in trial.questions if (m := _map_trial_question(q, rng, map_question)) is not None]
        return {"kind": "qcm", "questions": questions[:TRIAL_QUESTIONS[1]]} if questions else None

    def card() -> dict[str, Any] | None:
        mapped = _map_card(trial.card)
        return {"kind": "exercise_card", "card": mapped} if mapped else None

    first, second = (card, qcm) if trial.kind is TrialKind.EXERCISE_CARD else (qcm, card)
    return first() or second()


# ─── Étapes, vocabulaire, section ───────────────────────────────────────────────


def _map_steps(raw_steps: list[Any], rng: random.Random) -> tuple[list[dict[str, Any]], dict[int, int]]:
    """Mini-étapes exploitables et correspondance ancien numéro → nouveau (1-based), pour `widget_brief`."""
    steps: list[dict[str, Any]] = []
    renumbered: dict[int, int] = {}
    for old, step in enumerate(raw_steps, start=1):
        text = _clean(step.text, CYCLE_STEP_TEXT_MAX)
        support = _map_support(step.support, rng)
        if not text or support is None or len(steps) == CYCLE_STEPS_MAX:
            continue
        mapped: dict[str, Any] = {"text": text, "action": _clean(step.action, CYCLE_ACTION_MAX), "support": support}
        if prediction := _map_prediction(step.prediction, rng):
            mapped["prediction"] = prediction
        steps.append(mapped)
        renumbered[old] = len(steps)
    return steps, renumbered


def _map_vocabulary(items: list[Any]) -> list[dict[str, Any]]:
    vocabulary: list[dict[str, Any]] = []
    for item in items:
        term, plain = _clean(item.term, CYCLE_TERM_MAX), _clean(item.plain, CYCLE_PLAIN_MAX)
        if term and plain and term.casefold() not in {v["term"].casefold() for v in vocabulary} and len(vocabulary) < CYCLE_VOCABULARY_MAX:
            entry: dict[str, Any] = {"term": term, "plain": plain}
            if formula := _clean(item.formula_latex, CYCLE_FORMULA_MAX):
                entry["formula_latex"] = formula
            vocabulary.append(entry)
    return vocabulary


def _map_widget_brief(brief: WidgetBrief | None, renumbered: dict[int, int]) -> dict[str, Any] | None:
    if brief is None or brief.step_number not in renumbered:
        return None
    return _validated(ApiWidgetBrief, {"step_number": renumbered[brief.step_number], "brief": _clean(brief.brief, WIDGET_BRIEF_MAX)})


def _map_cycle(
    section: Section, map_question: MapQuestion, rng: random.Random
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[str]]:
    """`(cycle, widget_brief, raisons)` : `cycle` est None (avec au moins une raison) si le défi ou toutes les
    mini-étapes sont inexploitables, ou si l'API refuse le résultat."""
    cycle = section.cycle
    if cycle is None:
        return None, None, []
    challenge = _map_challenge(cycle.challenge, rng)
    steps, renumbered = _map_steps(cycle.steps, rng)
    reasons: list[str] = []
    if challenge is None:
        reasons.append("Défi inexploitable après nettoyage (3 à 5 options distinctes, bonne réponse valide, jamais toutes)")
    if not steps:
        reasons.append("Sous le capot : aucune mini-étape exploitable après nettoyage")
    if reasons:
        return None, None, reasons
    mapped = {
        "thread": _clean(cycle.thread, CYCLE_THREAD_MAX),
        "challenge": challenge,
        "steps": steps,
        "vocabulary": _map_vocabulary(cycle.vocabulary),
        "traps": _map_traps(cycle.traps, rng),
        "trial": _map_trial(cycle.trial, rng, map_question),
    }
    try:
        validated = ApiSectionCycle.model_validate(mapped).model_dump(mode="json", exclude_none=True)
    except ValidationError:
        return None, None, ["le cycle est refusé par la validation de l'API après nettoyage"]
    return validated, _map_widget_brief(section.widget_brief, renumbered), []


def map_section_cycle(
    section: Section, *, map_question: MapQuestion, rng: random.Random | None = None
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """`(cycle, widget_brief)` d'API d'une section générée au format cycle ; `(None, None)` sans cycle ou si le
    cycle est inexploitable (défi invalide ou aucune mini-étape valide). Ne lève jamais.

    `map_question` convertit une `QuizQuestion` Gemini en dict d'API (voir `course_generator._map_quiz_question`).
    `rng` : générateur du mélange (fixé par les tests ; par défaut, non déterministe)."""
    if section.cycle is None:
        return None, None
    cycle, brief, reasons = _map_cycle(section, map_question, rng or random.Random())
    if cycle is None:
        logger.warning("course_cycle_dropped", extra={"reasons": reasons})
    return cycle, brief


def _mapped_gaps(
    section: Section, profile: DepthProfile | None, map_question: MapQuestion
) -> list[str]:
    """Ce que le mapping écarte d'un cycle brut sans trou : un défi aux options en double, un support dont les
    cases sont vides, un cycle que l'API refuse... Le mélange n'a aucun effet sur ces comptes."""
    cycle = section.cycle
    assert cycle is not None
    mapped, _, reasons = _map_cycle(section, map_question, random.Random(0))
    if mapped is None:
        return reasons
    gaps: list[str] = []
    kept = len(mapped["steps"])
    gaps += [
        f"mini-étape {number} : support inexploitable après nettoyage (doublons, cases vides, valeurs non finies...)"
        for number in _dropped_steps(cycle.steps)
    ]
    low = profile.cycle.steps_min if profile is not None and profile.cycle is not None else 1
    if kept < low:
        gaps.append(f"Sous le capot : {kept} mini-étape(s) exploitable(s) après nettoyage pour {low} attendues au minimum")
    if not any(step["support"]["kind"] in _RECONSTRUCTION_VALUES for step in mapped["steps"]):
        gaps.append("aucune mini-étape de reconstruction exploitable après nettoyage (classer, ordonner, compléter ou relier)")
    if not mapped.get("vocabulary"):
        gaps.append("Le mot juste : aucun terme exploitable après nettoyage")
    if not mapped.get("traps"):
        gaps.append("Pièges classiques : aucune entrée exploitable après nettoyage (index et affirmations valides)")
    if not mapped.get("trial"):
        gaps.append("Épreuve : aucune question ou carte exploitable après nettoyage")
    return gaps


def _dropped_steps(raw_steps: list[CycleStep]) -> list[int]:
    """Numéros (1-based) des mini-étapes dont le mapping écarte le support, hors plafond de nombre d'étapes."""
    rng = random.Random(0)
    return [
        number for number, step in enumerate(raw_steps, start=1)
        if _clean(step.text, CYCLE_STEP_TEXT_MAX) and _map_support(step.support, rng) is None
    ]


def cycle_gaps(section: Section, profile: DepthProfile | None, map_question: MapQuestion) -> list[str]:
    """Trous BLOQUANTS du cycle d'une section : ceux du texte brut du modèle (`cycle_structural_gaps`) puis, s'il
    n'y en a aucun, ceux que le mapping vers l'API révèle (élément écarté, cycle refusé). Sans ce second temps, une
    section aurait un cycle « valide » à la génération mais vide une fois mappée, donc absente du cours.

    `map_question` : même convertisseur que `map_section_cycle`. Liste vide = conforme (ou pas de `cycle`)."""
    gaps = cycle_structural_gaps(section, profile)
    if gaps or section.cycle is None:
        return gaps
    return _mapped_gaps(section, profile, map_question)


def cycle_legacy_text(cycle: dict[str, Any]) -> tuple[str, str, str]:
    """Texte à plat `(pourquoi, quoi, comment)` d'un cycle d'API.

    Les champs historiques `pourquoi / quoi / comment` de `CourseSection` alimentent encore le podcast,
    l'assistant et la détection des sections de développement : une section au format cycle n'a pas de
    sous-sections, on y met donc le défi, le vocabulaire et le texte des mini-étapes."""
    vocabulary = " ; ".join(f"{v['term']} : {v['plain']}" for v in cycle.get("vocabulary", []))
    steps = " ".join(step["text"] for step in cycle.get("steps", []))
    return cycle["challenge"]["question"], vocabulary, steps


def with_cycle_minutes(section: CourseSection, depth: str) -> CourseSection:
    """Copie de `section` dont le cycle porte la durée estimée du mode `depth` (`estimate_minutes`, calculée
    côté serveur). Une section sans cycle (ancien format, introduction, résumé) est rendue telle quelle : pas de
    durée."""
    if section.cycle is None:
        return section
    low, high = estimate_minutes(depth, len(section.cycle.steps))
    return section.model_copy(
        update={"cycle": section.cycle.model_copy(update={"minutes_min": low, "minutes_max": high})}
    )
