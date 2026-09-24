"""Contrôle « visuel d'abord » des sections de développement.

Une section DEVELOPMENT doit contenir au moins un bloc non textuel (tableau, liste,
schéma, graphique, formule...) et ne garde que de courts blocs TEXT (≤ `TEXT_BLOCK_MAX_SENTENCES`).
"""

import re

from app.schemas.course_generation import BlockType, ContentBlock, Section, SectionType

TEXT_BLOCK_MAX_SENTENCES = 3

_SENTENCE_END_RE = re.compile(r"[.!?…]+(?:\s|$)")
_TEXTUAL = {BlockType.TEXT, BlockType.DEFINITION}
# TODO(supports visuels, Lot 2/3/5) : une fois un résolveur d'images réel branché
# (app/services/media/visual_resolver.py), remettre IMAGE ici. Tant qu'aucun résolveur n'existe
# (Lot 1), chaque bloc IMAGE est de toute façon retiré : l'exclure de cette règle ne fait ici que
# déclencher un appel Gemini de régénération payant (_repair_incomplete_sections dans
# course_plan_generator.py) sans jamais pouvoir aboutir à une image réellement affichée — pur
# surcoût de quota tant que ce n'est pas le cas.
_NOT_A_VISUAL_GUARANTEE = _TEXTUAL


def _blocks(section: Section) -> list[ContentBlock]:
    return [*section.blocks, *(b for sub in section.subsections for b in sub.blocks)]


def sentence_count(text: str) -> int:
    return len(_SENTENCE_END_RE.findall(text.strip())) or (1 if text.strip() else 0)


def visual_issues(section: Section) -> list[str]:
    """Problèmes « visuel d'abord » d'une section ; [] si conforme ou si ce n'est pas un développement."""
    if section.type is not SectionType.DEVELOPMENT:
        return []
    blocks = _blocks(section)
    issues: list[str] = []
    if not any(b.type not in _NOT_A_VISUAL_GUARANTEE for b in blocks):
        issues.append("aucun bloc visuel (TABLE, LIST, DIAGRAM, CHART, FORMULA...)")
    long_texts = sum(
        1 for b in blocks if b.type is BlockType.TEXT and b.text and sentence_count(b.text) > TEXT_BLOCK_MAX_SENTENCES
    )
    if long_texts:
        issues.append(f"{long_texts} bloc(s) TEXT de plus de {TEXT_BLOCK_MAX_SENTENCES} phrases")
    return issues
