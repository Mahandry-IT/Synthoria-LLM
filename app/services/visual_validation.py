"""Contrôle de forme (« visuel d'abord » + budget) des sections de développement.

Sans profil de mode (appel historique) : au moins un bloc non textuel et de courts blocs TEXT
(≤ `TEXT_BLOCK_MAX_SENTENCES`). Avec un profil (`app.services.course_depth`) : en plus, part
minimale de blocs non textuels, budget de mots de prose et nombre maximal de blocs.

Toutes ces règles sont « soft » (forme) : elles déclenchent une réparation bornée mais ne rendent
jamais une section incomplète (voir `course_plan_generator._finalize_incomplete`).

Une section au format « cycle » (`Section.cycle`) a ses propres règles de forme (`cycle_form_issues` : RG-3
alternance des supports, RG-4 longueurs). Elles sont aussi « soft », mais ne déclenchent jamais de réparation
à elles seules (voir `course_plan_generator.SectionIssues.form`).
"""

import re

from app.schemas.course_generation import (
    BlockType,
    ContentBlock,
    CycleChallenge,
    Section,
    SectionType,
    StepSupport,
    SupportKind,
)
from app.services.course_depth import DepthProfile
from app.services.cycle_validation import cycle_bounds_issues

TEXT_BLOCK_MAX_SENTENCES = 3

# RG-4 : un bloc de texte tient en 80 mots, une mini-étape vise environ 40 mots (1 à 3 phrases).
TEXT_WORDS_MAX = 80
STEP_WORDS_TARGET = 40
STEP_WORDS_SOFT_MAX = 55  # au-delà, la mini-étape s'éloigne nettement de la cible sans atteindre le plafond
STEP_SENTENCES_MAX = 3
# RG-4 : tableau de 5 lignes sur 3 à 4 colonnes, cellules de 8 mots ; RG-9 : correction du défi en une ligne.
TABLE_MAX_ROWS = 5
TABLE_COLS = (3, 4)
TABLE_CELL_WORDS_MAX = 8
ONE_LINE_WORDS_MAX = 25
VOCAB_PLAIN_WORDS_MAX = 20
ACTION_WORDS_MAX = 25
DIAGRAM_MAX_NODES = 8

_SENTENCE_END_RE = re.compile(r"[.!?…]+(?:\s|$)")
_WORD_RE = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*")
# Blocs textuels : ne comptent pas comme « non textuels » dans le ratio.
_TEXTUAL = {BlockType.TEXT, BlockType.DEFINITION, BlockType.CALLOUT}
# TODO(supports visuels) : les résolveurs d'images web et PDF étant branchés
# (app/services/media/visual_resolver.py), ajouter IMAGE ici — sa résolution peut échouer, il ne
# garantit donc pas un visuel. Non fait tant que le surcoût n'est pas mesuré : une section dont
# le seul visuel est une image déclencherait un appel Gemini de régénération payant
# (_repair_incomplete_sections dans course_plan_generator.py).
_NOT_A_VISUAL_GUARANTEE = _TEXTUAL


def _blocks(section: Section) -> list[ContentBlock]:
    return [*section.blocks, *(b for sub in section.subsections for b in sub.blocks)]


def sentence_count(text: str) -> int:
    return len(_SENTENCE_END_RE.findall(text.strip())) or (1 if text.strip() else 0)


def word_count(text: str | None) -> int:
    return len(_WORD_RE.findall(text or ""))


def _block_prose_words(block: ContentBlock) -> int:
    """Mots de prose d'un bloc : TEXT/DEFINITION/CALLOUT, éléments de LIST et exemple travaillé.
    Tableaux, formules, code, schémas, graphiques et images ne comptent pas."""
    if block.type in _TEXTUAL:
        return word_count(block.text)
    if block.type is BlockType.LIST:
        return sum(word_count(item) for item in block.list_items or [])
    if block.type is BlockType.WORKED_EXAMPLE and block.worked_example:
        example = block.worked_example
        return word_count(example.statement) + sum(word_count(s) for s in example.steps) + word_count(example.result)
    return 0


def prose_word_count(section: Section) -> int:
    """Mots de prose de tous les blocs d'une section (hors défi, exemple à trous et questions)."""
    return sum(_block_prose_words(b) for b in _blocks(section))


def visual_issues(section: Section, profile: DepthProfile | None = None) -> list[str]:
    """Problèmes de forme d'une section ; [] si conforme ou si ce n'est pas un développement.

    Messages chiffrés (« 412 mots pour un budget de 300 ») : ils sont recopiés tels quels dans le
    prompt de réparation pour guider la régénération.
    """
    if section.type is not SectionType.DEVELOPMENT:
        return []
    if section.cycle is not None:
        return cycle_form_issues(section, profile)
    blocks = _blocks(section)
    issues: list[str] = []
    non_textual = sum(1 for b in blocks if b.type not in _NOT_A_VISUAL_GUARANTEE)
    if not non_textual:
        issues.append("aucun bloc visuel (TABLE, LIST, DIAGRAM, CHART, FORMULA...)")
    elif profile is not None and non_textual / len(blocks) < profile.non_text_ratio:
        issues.append(
            f"{non_textual} bloc(s) non textuel(s) sur {len(blocks)} : au moins "
            f"{profile.non_text_ratio:.0%} attendus (remplace de la prose par des tableaux, listes, schémas...)"
        )
    long_texts = sum(
        1 for b in blocks if b.type is BlockType.TEXT and b.text and sentence_count(b.text) > TEXT_BLOCK_MAX_SENTENCES
    )
    if long_texts:
        issues.append(f"{long_texts} bloc(s) TEXT de plus de {TEXT_BLOCK_MAX_SENTENCES} phrases")
    if profile is not None:
        words = prose_word_count(section)
        if words > profile.max_prose_words:
            issues.append(f"{words} mots de prose pour un budget de {profile.max_prose_words}")
        if len(blocks) > profile.max_blocks:
            issues.append(f"{len(blocks)} blocs pour un maximum de {profile.max_blocks}")
    return issues


# ─── Format « cycle » ──────────────────────────────────────────────────────────────────────────────

_FORMULA_RE = re.compile(r"\$\$.+?\$\$|\$[^$\n]+\$", re.S)
_DISPLAY_FORMULA_RE = re.compile(r"\$\$.+?\$\$", re.S)
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$", re.M)
# Annonce du nombre de bonnes réponses : le nombre collé au mot « réponse(s) » (« une seule réponse », « toutes les
# bonnes réponses », « deux réponses correctes »). « deux ... bonnes pratiques » n'en est pas une.
_ANNOUNCES_COUNT_RE = re.compile(
    r"\b(?:une seule|toutes les|plusieurs|deux|trois|quatre)\s+(?:bonnes?\s+)?r[ée]ponses?\b", re.I
)
_MERMAID_STARTS = ("flowchart", "graph", "sequencediagram", "statediagram", "classdiagram")
_UNQUOTED_PAREN_LABEL_RE = re.compile(r"\[[^\]\"]*[()][^\]\"]*\]")
_MERMAID_NODE_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)\s*(?=[\[\(\{])")

# Bornes de longueur en caractères (contrat du cycle) : signalées, jamais bloquantes (l'API tronque).
_LENGTH_LIMITS = (
    ("fil rouge", 140), ("question du défi", 220), ("option du défi", 90), ("texte de mini-étape", 320),
    ("légende de support", 80), ("question de prédiction", 140), ("option de prédiction", 60),
    ("résultat de prédiction", 160), ("explication de piège", 220), ("affirmation de piège", 100),
)


def _step_support_issues(support: StepSupport, where: str) -> list[str]:
    issues: list[str] = []
    if support.kind is SupportKind.CHART and support.chart is not None and len(support.chart.steps) < 2:
        issues.append(f"{where} : graphique sans `steps` (2 à 4 attendues), il ne se parcourt pas")
    if support.kind is SupportKind.DIAGRAM and support.diagram is not None:
        source = support.diagram.mermaid.strip()
        if source and not source.lower().startswith(_MERMAID_STARTS):
            issues.append(f"{where} : schéma Mermaid qui ne commence pas par un type de diagramme valide")
        if _UNQUOTED_PAREN_LABEL_RE.search(source):
            issues.append(f"{where} : libellé Mermaid avec parenthèses sans guillemets (le schéma ne s'afficherait pas)")
        if len(set(_MERMAID_NODE_RE.findall(source))) > DIAGRAM_MAX_NODES:
            issues.append(f"{where} : schéma de plus de {DIAGRAM_MAX_NODES} nœuds")
    if support.kind in (SupportKind.TABLE, SupportKind.FILL_TABLE) and support.table is not None:
        table = support.table
        if len(table.rows) > TABLE_MAX_ROWS:
            issues.append(f"{where} : tableau de {len(table.rows)} lignes (plafond {TABLE_MAX_ROWS})")
        if not TABLE_COLS[0] <= len(table.headers) <= TABLE_COLS[1]:
            issues.append(f"{where} : tableau de {len(table.headers)} colonnes ({TABLE_COLS[0]} à {TABLE_COLS[1]} attendues)")
        if any(len(row) != len(table.headers) for row in table.rows):
            issues.append(f"{where} : ligne de tableau de longueur différente des en-têtes")
        if any(word_count(cell) > TABLE_CELL_WORDS_MAX for row in table.rows for cell in row):
            issues.append(f"{where} : cellule de tableau de plus de {TABLE_CELL_WORDS_MAX} mots")
    return issues


def _step_text_issues(text: str, where: str) -> list[str]:
    issues: list[str] = []
    words = word_count(text)
    if words > TEXT_WORDS_MAX:
        issues.append(f"{where} : {words} mots pour un plafond de {TEXT_WORDS_MAX}")
    elif words > STEP_WORDS_SOFT_MAX:
        issues.append(f"{where} : {words} mots, environ {STEP_WORDS_TARGET} visés")
    if sentence_count(text) > STEP_SENTENCES_MAX:
        issues.append(f"{where} : plus de {STEP_SENTENCES_MAX} phrases")
    if len(_FORMULA_RE.findall(text)) > 1:
        issues.append(f"{where} : plus d'une formule dans le texte (une seule par mini-étape)")
    if _DISPLAY_FORMULA_RE.search(text) or len(_TABLE_ROW_RE.findall(text)) >= 2:
        issues.append(f"{where} : le texte mélange prose et formule en bloc ou tableau (chaque chose dans son bloc)")
    return issues


def _challenge_issues(challenge: CycleChallenge) -> list[str]:
    issues: list[str] = []
    if challenge.feedback.strip() and (
        word_count(challenge.feedback) > ONE_LINE_WORDS_MAX or sentence_count(challenge.feedback) > 1
    ):
        issues.append("Défi : correction de plus d'une ligne (25 mots, une phrase)")
    if _ANNOUNCES_COUNT_RE.search(challenge.question):
        issues.append("Défi : la question annonce le nombre de bonnes réponses")
    return issues


def _length_issues(section: Section) -> list[str]:
    """Bornes de longueur en caractères du contrat du cycle."""
    cycle = section.cycle
    assert cycle is not None
    values: list[tuple[str, str]] = [("fil rouge", cycle.thread), ("question du défi", cycle.challenge.question)]
    values += [("option du défi", o) for o in cycle.challenge.options]
    for step in cycle.steps:
        values += [("texte de mini-étape", step.text), ("légende de support", step.support.caption)]
        if step.prediction is not None:
            values += [("question de prédiction", step.prediction.question), ("résultat de prédiction", step.prediction.outcome)]
            values += [("option de prédiction", o) for o in step.prediction.options]
    for entry in cycle.traps.entries:
        values += [("explication de piège", entry.explanation)]
        values += [("affirmation de piège", s) for s in [entry.claim, *entry.statements, *entry.rewrite_options]]
    limits = dict(_LENGTH_LIMITS)
    over = sorted({label for label, text in values if len(text) > limits[label]})
    return [f"{label} : plus de {limits[label]} caractères" for label in over]


def cycle_form_issues(section: Section, profile: DepthProfile | None = None) -> list[str]:
    """Problèmes de FORME (soft) du `cycle` d'une section : RG-3 (jamais deux supports du même type d'affilée),
    RG-4 (80 mots par bloc de texte, environ 40 par mini-étape ; tableau de 5 lignes sur 3 à 4 colonnes aux
    cellules de 8 mots ; une formule par mini-étape ; pas de mélange texte / tableau / formule), plus les
    plafonds du défi, du vocabulaire et des prédictions. [] sans `cycle`.

    Jamais bloquant : la structure (cinq temps présents, index valides) est contrôlée à part par
    `cycle_validation.cycle_structural_gaps`."""
    cycle = section.cycle
    if cycle is None:
        return []
    issues = _challenge_issues(cycle.challenge)
    previous: SupportKind | None = None
    predictions: list[int] = []
    for number, step in enumerate(cycle.steps, start=1):
        where = f"mini-étape {number}"
        issues += _step_text_issues(step.text, where)
        if step.action.strip() and word_count(step.action) > ACTION_WORDS_MAX:
            issues.append(f"{where} : consigne de plus de {ACTION_WORDS_MAX} mots")
        if previous is not None and previous is step.support.kind:
            issues.append(f"{where} : même type de support ({previous.value}) que la mini-étape précédente")
        previous = step.support.kind
        issues += _step_support_issues(step.support, where)
        if step.prediction is not None:
            predictions.append(number)
    if any(b - a == 1 for a, b in zip(predictions, predictions[1:])):
        issues.append("prédictions sur des mini-étapes consécutives")
    if any(word_count(v.plain) > VOCAB_PLAIN_WORDS_MAX for v in cycle.vocabulary):
        issues.append(f"Le mot juste : définition de plus de {VOCAB_PLAIN_WORDS_MAX} mots")
    if profile is not None and profile.cycle is not None:
        issues += _profile_issues(section, profile, len(predictions))
    return [*issues, *cycle_bounds_issues(section), *_length_issues(section)]


def _profile_issues(section: Section, profile: DepthProfile, predictions: int) -> list[str]:
    """Écarts au profil du mode : nombre de prédictions, de termes, de puces, forme des pièges."""
    cycle, rules = section.cycle, profile.cycle
    assert cycle is not None and rules is not None
    issues: list[str] = []
    if not rules.predictions_min <= predictions <= rules.predictions_max:
        issues.append(
            f"{predictions} prédiction(s) pour {rules.predictions_min} à {rules.predictions_max} attendues dans la section"
        )
    if not rules.vocab_min <= len(cycle.vocabulary) <= rules.vocab_max:
        issues.append(f"Le mot juste : {len(cycle.vocabulary)} terme(s) pour {rules.vocab_min} à {rules.vocab_max} attendus")
    if cycle.traps.mode.value != rules.traps_mode:
        issues.append(f"Pièges : mode `{cycle.traps.mode.value}` au lieu de `{rules.traps_mode}`")
    elif not rules.traps_min <= len(cycle.traps.entries) <= rules.traps_max:
        issues.append(f"Pièges : {len(cycle.traps.entries)} entrée(s) pour {rules.traps_min} à {rules.traps_max} attendues")
    if len(section.key_takeaways) != rules.takeaways:
        issues.append(f"{len(section.key_takeaways)} `key_takeaways` pour {rules.takeaways} attendues")
    return issues

