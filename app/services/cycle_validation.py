"""Contrôle de structure BLOQUANT du `cycle` d'une section de développement.

Remplace, pour une section qui porte un `cycle`, le contrôle par titres Pourquoi / Quoi / Comment
(`course_plan_generator._structural_gaps`) : les cinq temps (Défi, Sous le capot, Le mot juste, Les pièges
classiques, Épreuve) doivent être présents et exploitables. Un trou ici rend la section « incomplète » après
réparation, comme aujourd'hui ; les écarts de forme, eux, relèvent de `visual_validation` (soft).

Ce module contient aussi le contrôle SOFT « la réponse d'une prédiction apparaît dans le texte qui la
précède » (`prediction_reveal_issues`) : heuristique lexicale, jamais une cause de réparation à elle seule.

Les nombres attendus (mini-étapes) viennent du profil du mode (`course_depth.CycleProfile`), jamais d'ici.
"""

import math
import re
import unicodedata

from app.schemas.course_generation import (
    RECONSTRUCTION_KINDS,
    CycleChallenge,
    CycleStep,
    CycleTraps,
    CycleTrial,
    PredictionKind,
    QuizKind,
    Section,
    StepPrediction,
    StepSupport,
    SupportKind,
    TableData,
    TrapItem,
    TrapMode,
    TrialKind,
    quiz_question_issue,
)
from app.services.course_depth import PROFILES, DepthProfile

# Bornes des listes d'un cycle (le schéma envoyé à Gemini n'en porte aucune : voir `CycleSectionsBatchSchema`).
# Sous le minimum : BLOQUANT (réparation). Au-dessus du maximum : SOFT (`cycle_bounds_issues`), le mapping
# tronque (`cycle_mapping`), jamais de réparation pour un excès.
CHALLENGE_OPTIONS = (3, 5)
PREDICTION_OPTIONS = (2, 4)
FILL_TABLE_BLANKS = (1, 4)
CLASSIFY_CATEGORIES = (2, 4)
CLASSIFY_CASES = (3, 6)
ORDER_ITEMS = (3, 6)
MATCH_PAIRS = (3, 5)
SPOT_STATEMENTS = (3, 4)
REWRITE_OPTIONS = (2, 3)
TRIAL_QUESTIONS = (1, 2)
TRIAL_OPTIONS_MAX = 5
# Une prédiction `value` n'est jamais plus tolérante que 10 % de la valeur attendue (RG-16).
PREDICTION_TOLERANCE_MAX_RATIO = 0.1


def _blank(value: str | None) -> bool:
    return not (value or "").strip()


def _below(count: int, bounds: tuple[int, int]) -> bool:
    return count < bounds[0]


def _above(count: int, bounds: tuple[int, int]) -> bool:
    return count > bounds[1]


def _distinct(items: list[str]) -> int:
    """Nombre de textes non vides et distincts (sans tenir compte de la casse ni des espaces de bord)."""
    return len({item.strip().casefold() for item in items if not _blank(item)})


def _steps_bounds(profile: DepthProfile | None) -> tuple[int, int]:
    """Mini-étapes acceptées ; sans profil (appel historique), la plage couvre tous les modes."""
    if profile is not None and profile.cycle is not None:
        return profile.cycle.steps_min, profile.cycle.steps_max
    cycles = [p.cycle for p in PROFILES.values() if p.cycle is not None]
    return min(c.steps_min for c in cycles), max(c.steps_max for c in cycles)


def _valid_index(index: int | None, size: int) -> bool:
    return index is not None and 0 <= index < size


def _cell(table: TableData, row: int | None, col: int | None) -> str | None:
    """Texte de la case `(row, col)` d'un tableau, ou None si elle n'existe pas."""
    if row is None or col is None or not _valid_index(row, len(table.rows)) or not _valid_index(col, len(table.headers)):
        return None
    cells = table.rows[row]
    return cells[col] if col < len(cells) else ""


# ─── Supports ──────────────────────────────────────────────────────────────────


def _fill_table_gaps(support: StepSupport, table: TableData) -> list[str]:
    if _below(len(support.blanks), FILL_TABLE_BLANKS):
        return ["tableau à compléter : 1 à 4 cases masquées (`blanks`) attendues"]
    cells = [_cell(table, b.row, b.col) for b in support.blanks]
    if any(cell is None for cell in cells):
        return ["tableau à compléter : case masquée hors du tableau"]
    if all(_blank(cell) for cell in cells):
        return ["tableau à compléter : les cases masquées sont vides (la réponse est le texte de la case)"]
    return []


def _support_gaps(support: StepSupport) -> list[str]:
    kind = support.kind
    if kind is None:
        return ["support sans `kind` valide (chart, diagram, table, fill_table, classify, order ou match)"]
    if kind is SupportKind.CHART:
        chart = support.chart
        if chart is None or not chart.labels or not chart.series:
            return ["graphique vide"]
    elif kind is SupportKind.DIAGRAM:
        if support.diagram is None or _blank(support.diagram.mermaid):
            return ["schéma vide"]
    elif kind in (SupportKind.TABLE, SupportKind.FILL_TABLE):
        table = support.table
        if table is None or not table.headers or not any(any(not _blank(c) for c in row) for row in table.rows):
            return ["tableau vide ou absent (`table` obligatoire pour `table` et `fill_table`)"]
        if kind is SupportKind.FILL_TABLE:
            return _fill_table_gaps(support, table)
    elif kind is SupportKind.CLASSIFY:
        if _below(len(support.categories), CLASSIFY_CATEGORIES) or _below(len(support.cases), CLASSIFY_CASES):
            return ["classement : 2 à 4 catégories et 3 à 6 cas attendus"]
        if any(not _valid_index(c.category_index, len(support.categories)) for c in support.cases):
            return ["classement : catégorie d'un cas hors des catégories"]
    elif kind is SupportKind.ORDER:
        if _below(len(support.order_items), ORDER_ITEMS):
            return ["remise dans l'ordre : 3 à 6 éléments attendus (`order_items`)"]
    elif kind is SupportKind.MATCH:
        if _below(len(support.match_pairs), MATCH_PAIRS):
            return ["association : 3 à 5 paires attendues (`match_pairs`)"]
        rights = [p.right.strip().casefold() for p in support.match_pairs]
        if len(set(rights)) != len(rights):
            return ["association : les valeurs `right` doivent être toutes distinctes"]
    return []


def _prediction_gaps(prediction: StepPrediction) -> list[str]:
    if prediction.kind is None:
        return ["prédiction sans `kind` valide (choice ou value)"]
    gaps: list[str] = []
    if _blank(prediction.question) or _blank(prediction.outcome):
        gaps.append("prédiction sans `question` ou sans `outcome`")
    if prediction.kind is PredictionKind.CHOICE:
        if _below(len(prediction.options), PREDICTION_OPTIONS):
            gaps.append("prédiction `choice` : 2 à 4 options attendues")
        elif not _valid_index(prediction.correct_index, len(prediction.options)):
            gaps.append("prédiction `choice` : `correct_index` hors des options")
    elif prediction.expected_value is None or not math.isfinite(prediction.expected_value):
        gaps.append("prédiction `value` : `expected_value` obligatoire")
    return gaps


# ─── Les cinq temps ─────────────────────────────────────────────────────────────


def _challenge_gaps(challenge: CycleChallenge) -> list[str]:
    gaps: list[str] = []
    if _blank(challenge.question):
        gaps.append("Défi : question vide")
    if _distinct(challenge.options) < CHALLENGE_OPTIONS[0]:
        gaps.append("Défi : 3 à 5 options distinctes et non vides attendues")
    ids = challenge.correct_indices
    if not ids or any(not _valid_index(i, len(challenge.options)) for i in ids):
        gaps.append("Défi : `correct_indices` doit nommer au moins une option existante")
    elif len(set(ids)) >= len(challenge.options):
        gaps.append("Défi : `correct_indices` ne doit jamais désigner toutes les options")
    return gaps


def _trap_entry_gap(entry: TrapItem, mode: TrapMode) -> bool:
    """Vrai si une entrée de pièges est inexploitable pour le `mode` des pièges."""
    if _blank(entry.explanation):
        return True
    if mode is TrapMode.CARDS:
        return _blank(entry.claim)
    if _below(len(entry.statements), SPOT_STATEMENTS) or not _valid_index(entry.wrong_index, len(entry.statements)):
        return True
    return mode is TrapMode.SPOT_REWRITE and (
        _below(len(entry.rewrite_options), REWRITE_OPTIONS)
        or not _valid_index(entry.correct_rewrite_index, len(entry.rewrite_options))
    )


def _traps_gaps(traps: CycleTraps) -> list[str]:
    if not traps.entries:
        return ["Pièges classiques : aucune entrée"]
    if any(_trap_entry_gap(e, traps.mode) for e in traps.entries):
        return [f"Pièges classiques (mode `{traps.mode.value}`) : entrée incomplète ou à index invalide"]
    return []


def _trial_gaps(trial: CycleTrial) -> list[str]:
    if trial.kind is TrialKind.QCM:
        if not trial.questions or any(quiz_question_issue(q) or q.kind is QuizKind.MATCHING for q in trial.questions):
            return ["Épreuve `qcm` : 1 à 2 questions valides attendues (`choice` ou `numeric`)"]
    elif trial.card is None or _blank(trial.card.statement) or not any(not _blank(s) for s in trial.card.steps):
        return ["Épreuve `exercise_card` : énoncé et étapes de correction obligatoires"]
    return []


def _widget_brief_gaps(section: Section, steps: int) -> list[str]:
    brief = section.widget_brief
    if brief is None or (brief.step_number is not None and 1 <= brief.step_number <= steps):
        return []
    return [f"`widget_brief` : `step_number` doit désigner une mini-étape existante (1 à {steps})"]


def cycle_structural_gaps(section: Section, profile: DepthProfile | None = None) -> list[str]:
    """Éléments manquants ou inutilisables du `cycle` d'une section (liste vide = conforme ; une section sans
    `cycle` n'est pas concernée et rend []). Messages courts, recopiés dans le prompt de réparation.

    Ne contrôle que le texte brut du modèle. Ce que le mapping écarte ou tronque ensuite (doublons, cases vides...)
    est contrôlé par `cycle_mapping.mapped_cycle_gaps`."""
    cycle = section.cycle
    if cycle is None:
        return []
    gaps = _challenge_gaps(cycle.challenge)

    low, high = _steps_bounds(profile)
    if not low <= len(cycle.steps) <= high:
        gaps.append(f"Sous le capot : {len(cycle.steps)} mini-étape(s) pour {low} à {high} attendues")
    for number, step in enumerate(cycle.steps, start=1):
        if _blank(step.text):
            gaps.append(f"mini-étape {number} : texte vide")
        gaps.extend(f"mini-étape {number} : {gap}" for gap in _support_gaps(step.support))
        if step.prediction is not None:
            gaps.extend(f"mini-étape {number} : {gap}" for gap in _prediction_gaps(step.prediction))

    if not any(not _blank(v.term) and not _blank(v.plain) for v in cycle.vocabulary):
        gaps.append("Le mot juste : aucun terme exploitable (`term` + `plain`)")
    gaps.extend(_traps_gaps(cycle.traps))
    gaps.extend(_trial_gaps(cycle.trial))
    gaps.extend(_widget_brief_gaps(section, len(cycle.steps)))

    # RG-15 : une section n'est jamais faite de texte seul. Au moins une mini-étape demande TOUJOURS une
    # reconstruction (classer, ordonner, compléter un tableau, relier), widget ou non : le widget n'est jamais
    # garanti (plafonds du cours, refus à la relecture, régénération et ajout de section sans widget).
    if not any(s.support.kind in RECONSTRUCTION_KINDS for s in cycle.steps):
        gaps.append("aucune mini-étape de reconstruction (classer, ordonner, compléter un tableau ou relier)")
    return gaps


# ─── Contrôle SOFT : dépassements de maximum (le mapping tronque) ─────────────────────────────────────────


def _tolerance_issues(prediction: StepPrediction) -> list[str]:
    value, tolerance = prediction.expected_value, prediction.tolerance
    if prediction.kind is not PredictionKind.VALUE or value is None or not math.isfinite(value):
        return []
    if tolerance is None or not math.isfinite(tolerance) or tolerance <= 0:
        return []
    if float(value).is_integer():
        return ["prédiction `value` : tolérance non nulle pour un entier ou un dénombrement (0 attendu)"]
    if tolerance > abs(value) * PREDICTION_TOLERANCE_MAX_RATIO:
        return ["prédiction `value` : tolérance supérieure à 10 % de la valeur attendue"]
    return []


def _step_bounds_issues(number: int, step: CycleStep) -> list[str]:
    where, support = f"mini-étape {number}", step.support
    issues: list[str] = []
    if _above(len(support.blanks), FILL_TABLE_BLANKS):
        issues.append(f"{where} : plus de {FILL_TABLE_BLANKS[1]} cases masquées")
    if _above(len(support.categories), CLASSIFY_CATEGORIES):
        issues.append(f"{where} : plus de {CLASSIFY_CATEGORIES[1]} catégories")
    if _above(len(support.cases), CLASSIFY_CASES):
        issues.append(f"{where} : plus de {CLASSIFY_CASES[1]} cas à classer")
    if _above(len(support.order_items), ORDER_ITEMS):
        issues.append(f"{where} : plus de {ORDER_ITEMS[1]} éléments à remettre dans l'ordre")
    if _above(len(support.match_pairs), MATCH_PAIRS):
        issues.append(f"{where} : plus de {MATCH_PAIRS[1]} paires à relier")
    if step.prediction is not None:
        if _above(len(step.prediction.options), PREDICTION_OPTIONS):
            issues.append(f"{where} : prédiction avec plus de {PREDICTION_OPTIONS[1]} options")
        issues.extend(f"{where} : {issue}" for issue in _tolerance_issues(step.prediction))
    return issues


def cycle_bounds_issues(section: Section) -> list[str]:
    """Dépassements de maximum du `cycle` (soft, jamais de réparation : le mapping tronque) et tolérances de
    prédiction `value` hors règle. [] sans `cycle`."""
    cycle = section.cycle
    if cycle is None:
        return []
    challenge = cycle.challenge
    issues: list[str] = []
    if _above(len(challenge.options), CHALLENGE_OPTIONS):
        issues.append(f"Défi : plus de {CHALLENGE_OPTIONS[1]} options")
    if len(set(challenge.correct_indices)) != len(challenge.correct_indices):
        issues.append("Défi : `correct_indices` en doublon")
    for number, step in enumerate(cycle.steps, start=1):
        issues += _step_bounds_issues(number, step)
    for entry in cycle.traps.entries:
        if _above(len(entry.statements), SPOT_STATEMENTS):
            issues.append(f"Pièges : plus de {SPOT_STATEMENTS[1]} affirmations dans une entrée")
        if _above(len(entry.rewrite_options), REWRITE_OPTIONS):
            issues.append(f"Pièges : plus de {REWRITE_OPTIONS[1]} reformulations dans une entrée")
    if _above(len(cycle.trial.questions), TRIAL_QUESTIONS):
        issues.append(f"Épreuve : plus de {TRIAL_QUESTIONS[1]} questions")
    return list(dict.fromkeys(issues))


def cycle_has_content(section: Section) -> bool:
    """Vrai si le cycle porte un défi et au moins une mini-étape avec du texte (équivalent de `_has_content`)."""
    cycle = section.cycle
    return bool(cycle and not _blank(cycle.challenge.question) and any(not _blank(s.text) for s in cycle.steps))


# ─── Contrôle soft : une prédiction ne doit pas être déjà répondue ─────────────────────────────────

_STOP_WORDS = frozenset({
    "les", "des", "une", "est", "que", "qui", "pour", "dans", "avec", "par", "sur", "the", "and", "are", "was",
    "has", "have", "not", "pas", "son", "ses", "aux", "elle", "ils", "vous", "nous",
})
_REVEAL_COVERAGE_MIN = 0.5
# Séparateur de milliers : espace (normale, insécable ou fine) suivi d'EXACTEMENT trois chiffres. « 3 4 » reste
# deux nombres (3 et 4), « 12 345 » n'en fait qu'un.
_NUMBER_RE = re.compile(r"(?<![\w.])-?(?:\d{1,3}(?:[ \u00a0\u202f]\d{3}(?!\d))+|\d+)(?:[.,]\d+)?")


def _normalize(text: str | None) -> str:
    decomposed = unicodedata.normalize("NFKD", (text or "").lower())
    plain = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", plain).strip()


def _numbers(text: str | None) -> list[float]:
    values: list[float] = []
    for match in _NUMBER_RE.findall(text or ""):
        try:
            values.append(float(re.sub(r"[\s  ]", "", match).replace(",", ".")))
        except ValueError:
            continue
    return values


def _coverage(option: str, text: str) -> tuple[float, int]:
    """Part des mots significatifs de `option` présents dans `text` (déjà normalisé), et leur nombre."""
    words = [w for w in _normalize(option).split() if (len(w) >= 4 or w.isdigit()) and w not in _STOP_WORDS]
    if not words:
        return 0.0, 0
    return sum(1 for w in words if f" {w} " in f" {text} ") / len(words), len(words)


def reveals_answer(prediction: StepPrediction, text: str) -> bool:
    """Heuristique lexicale : la réponse de `prediction` se lit-elle dans `text` ?

    `choice` : la bonne option y figure telle quelle, ou la plupart de ses mots significatifs y figurent (au
    moins 50 %, mieux que chaque distracteur). `value` : le nombre attendu y figure. Ne voit pas les
    révélations purement sémantiques : c'est un signal, pas une preuve."""
    if prediction.kind is None:
        return False
    if prediction.kind is PredictionKind.CHOICE:
        index = prediction.correct_index
        if not _valid_index(index, len(prediction.options)):
            return False
        haystack = _normalize(text)
        right = prediction.options[index]
        normalized = _normalize(right)
        if len(normalized) >= 4 and f" {normalized} " in f" {haystack} ":
            return True
        coverage, size = _coverage(right, haystack)
        others = [_coverage(o, haystack)[0] for i, o in enumerate(prediction.options) if i != index]
        return coverage >= _REVEAL_COVERAGE_MIN and (size >= 2 or coverage == 1.0) and coverage > max(others, default=0.0)
    value = prediction.expected_value
    return value is not None and any(abs(n - value) < 1e-9 for n in _numbers(text))


def prediction_reveal_issues(section: Section) -> list[str]:
    """Soft : la réponse d'une prédiction apparaît dans le texte de sa propre mini-étape ou des précédentes.
    L'apprenant prédit avant de manipuler : si le texte donne déjà la réponse, la prédiction ne sert à rien."""
    if section.cycle is None:
        return []
    steps = section.cycle.steps
    issues: list[str] = []
    for number, step in enumerate(steps, start=1):
        prediction = step.prediction
        if prediction is None:
            continue
        if reveals_answer(prediction, step.text):
            issues.append(f"mini-étape {number} : la réponse de la prédiction apparaît dans le texte de l'étape")
        earlier = " ".join(s.text for s in steps[: number - 1])
        # Un petit entier (0 à 9) se retrouve forcément dans un texte antérieur : pas un signal.
        small_integer = (
            prediction.kind is PredictionKind.VALUE and prediction.expected_value is not None
            and abs(prediction.expected_value) < 10 and float(prediction.expected_value).is_integer()
        )
        if earlier and not small_integer and reveals_answer(prediction, earlier):
            issues.append(f"mini-étape {number} : la réponse de la prédiction apparaît dans les étapes précédentes")
    return issues


def cycle_text(section: Section) -> str:
    """Tout le texte du cycle (défi, mini-étapes et leurs supports, vocabulaire, pièges, épreuve), pour la
    vérification déterministe de couverture des sous-thèmes (`covered_subtopics` absent ou contredit)."""
    cycle = section.cycle
    if cycle is None:
        return ""
    parts: list[str] = [cycle.thread, cycle.challenge.question, *cycle.challenge.options, cycle.challenge.feedback]
    for step in cycle.steps:
        support = step.support
        parts += [step.text, step.action, support.caption, support.order_criterion, *support.categories, *support.order_items]
        parts += [*(c.label for c in support.cases), *(f"{p.left} {p.right}" for p in support.match_pairs)]
        if support.table is not None:
            parts += [support.table.caption, *support.table.headers, *(cell for row in support.table.rows for cell in row)]
        if support.chart is not None:
            parts += [support.chart.caption, *support.chart.labels]
        if support.diagram is not None:
            parts.append(support.diagram.caption)
        if step.prediction is not None:
            parts += [step.prediction.question, step.prediction.outcome, *step.prediction.options]
    parts += [f"{v.term} {v.plain}" for v in cycle.vocabulary]
    for entry in cycle.traps.entries:
        parts += [entry.claim, entry.explanation, *entry.statements, *entry.rewrite_options]
    for question in cycle.trial.questions:
        parts += [question.question, question.explanation, *question.choices]
    if cycle.trial.card is not None:
        parts += [cycle.trial.card.statement, *cycle.trial.card.steps, cycle.trial.card.result]
    return " ".join(p for p in parts if p)
