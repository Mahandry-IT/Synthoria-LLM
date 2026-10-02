"""Contrôle de forme (« visuel d'abord » + budget) des sections de développement.

Sans profil de mode (appel historique) : au moins un bloc non textuel et de courts blocs TEXT
(≤ `TEXT_BLOCK_MAX_SENTENCES`). Avec un profil (`app.services.course_depth`) : en plus, part
minimale de blocs non textuels, budget de mots de prose et nombre maximal de blocs.

Toutes ces règles sont « soft » (forme) : elles déclenchent une réparation bornée mais ne rendent
jamais une section incomplète (voir `course_plan_generator._finalize_incomplete`).
"""

import re

from app.schemas.course_generation import BlockType, ContentBlock, Section, SectionType
from app.services.course_depth import DepthProfile

TEXT_BLOCK_MAX_SENTENCES = 3

_SENTENCE_END_RE = re.compile(r"[.!?…]+(?:\s|$)")
_WORD_RE = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*")
# Blocs textuels : ne comptent pas comme « non textuels » dans le ratio.
_TEXTUAL = {BlockType.TEXT, BlockType.DEFINITION, BlockType.CALLOUT}
# TODO(supports visuels, Lot 2/3/5) : une fois un résolveur d'images réel branché
# (app/services/media/visual_resolver.py), ajouter IMAGE ici. Tant qu'aucun résolveur n'existe
# (Lot 1), chaque bloc IMAGE est de toute façon retiré : le compter comme textuel ne ferait que
# déclencher un appel Gemini de régénération payant (_repair_incomplete_sections dans
# course_plan_generator.py) sans jamais pouvoir aboutir à une image réellement affichée — pur
# surcoût de quota tant que ce n'est pas le cas.
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
