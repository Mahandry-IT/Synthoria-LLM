"""Contrôle « visuel d'abord » des sections de développement.

Une section DEVELOPMENT doit contenir au moins un bloc non textuel (tableau, liste,
schéma, graphique, formule...) et ne garde que de courts blocs TEXT (≤ `TEXT_BLOCK_MAX_SENTENCES`).
"""

import re

from app.schemas.course_generation import BlockType, ContentBlock, Section, SectionType

TEXT_BLOCK_MAX_SENTENCES = 3

_SENTENCE_END_RE = re.compile(r"[.!?…]+(?:\s|$)")
_TEXTUAL = {BlockType.TEXT, BlockType.DEFINITION}
# IMAGE ne compte jamais comme le bloc visuel de la règle : sa résolution (app/services/media/
# visual_resolver.py) peut échouer et retirer le bloc après coup — la section ne doit pas dépendre
# d'un visuel qui pourrait disparaître silencieusement.
_NOT_A_VISUAL_GUARANTEE = _TEXTUAL | {BlockType.IMAGE}


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
