"""Fidélité plan → cours : sous-section vide, sous-thème du plan non traité, absence de visuel.

Couvre la boucle `_repair_incomplete_sections` (bornée à `_MAX_REPAIR_ATTEMPTS`) et le filet de
sécurité `_finalize_incomplete`, qui remplacent l'ancien `_fill_subtopic_gaps`/`_enforce_visual_first`
(un seul essai chacun, et un `covered_subtopics` vide désactivait purement le contrôle des
sous-thèmes — voir la régression identifiée sur un cours « statistiques et probabilités » limité à
88 % de couverture du plan).
"""

from unittest.mock import AsyncMock

import pytest

from app.api.schemas import ApiPlannedSection
from app.core.exceptions import GeminiUnavailableError
from app.schemas.course_generation import Section
from app.services.course_plan_generator import (
    _completeness_issues,
    _finalize_incomplete,
    _missing_subtopics,
    _repair_incomplete_sections,
    _structural_gaps,
)
from app.services.course_generator import INCOMPLETE_SECTION_NOTICE, is_incomplete_section


def _full_section(
    *, title: str = "S", covered_subtopics: list[str] | None = None,
    pourquoi: str = "Ça sert à comprendre.", quoi: str = "C'est une notion.", comment: str = "On l'applique ainsi.",
    extra_comment_blocks: list[dict] | None = None, subsection_titles: tuple[str, ...] = ("Pourquoi", "Quoi", "Comment"),
) -> Section:
    """Section DEVELOPMENT avec ses 3 sous-sections remplies (le cas conforme aux instructions)."""
    texts = {"Pourquoi": pourquoi, "Quoi": quoi, "Comment": comment}
    subsections = []
    for name in subsection_titles:
        blocks = [{"type": "text", "text": texts[name]}]
        if name == "Comment" and extra_comment_blocks:
            blocks += extra_comment_blocks
        subsections.append({"title": name, "blocks": blocks})
    return Section.model_validate({
        "type": "development", "title": title, "blocks": [], "subsections": subsections,
        "covered_subtopics": covered_subtopics or [],
    })


def _planned(title: str = "S", subtopics: list[str] | None = None) -> ApiPlannedSection:
    return ApiPlannedSection(type="development", title=title, objective="", subtopics=subtopics or [], order=1)


# ─── Lot A : sous-sections structurellement vides ────────────────────────────


def test_structural_gaps_detects_empty_subsections():
    section = _full_section(subsection_titles=("Pourquoi",))  # Quoi et Comment absentes
    assert _structural_gaps(section) == ["Quoi", "Comment"]


def test_structural_gaps_empty_when_all_three_filled():
    assert _structural_gaps(_full_section()) == []


def test_structural_gaps_flags_subsection_present_but_with_no_blocks():
    section = Section.model_validate({
        "type": "development", "title": "S", "blocks": [],
        "subsections": [
            {"title": "Pourquoi", "blocks": [{"type": "text", "text": "x"}]},
            {"title": "Quoi", "blocks": []},  # sous-section déclarée mais vide
            {"title": "Comment", "blocks": [{"type": "text", "text": "y"}]},
        ],
    })
    assert _structural_gaps(section) == ["Quoi"]


# ─── Lot B : sous-thèmes non traités, avec repli déterministe ───────────────


def test_missing_subtopics_uses_declared_coverage_when_present():
    section = _full_section(covered_subtopics=["Moyenne", "Médiane"])
    planned = _planned(subtopics=["Moyenne", "Médiane", "Variance"])
    assert _missing_subtopics(section, planned) == ["Variance"]


def test_missing_subtopics_falls_back_to_text_when_not_declared():
    """Avant ce correctif : covered_subtopics vide désactivait purement le contrôle (retournait [])
    même quand la section avait du contenu — laissant passer des sous-thèmes jamais traités."""
    section = _full_section(
        covered_subtopics=[],
        comment="On calcule la moyenne en sommant les valeurs puis en divisant par leur nombre.",
    )
    planned = _planned(subtopics=["Moyenne", "Variance et écart-type"])
    missing = _missing_subtopics(section, planned)
    assert missing == ["Variance et écart-type"]  # "moyenne" trouvé dans le texte, "variance" non


def test_missing_subtopics_returns_empty_for_content_free_section():
    """Une section sans aucun contenu (échec de génération) ne déclenche pas ce contrôle coûteux :
    elle est de toute façon gérée comme une section de repli ailleurs."""
    empty = Section.model_validate({"type": "development", "title": "S", "blocks": [], "subsections": []})
    planned = _planned(subtopics=["Moyenne"])
    assert _missing_subtopics(empty, planned) == []


# ─── Complétude fusionnée (structure + sous-thèmes + visuel) ────────────────


def test_completeness_issues_combines_all_checks():
    section = _full_section(subsection_titles=("Pourquoi", "Quoi"), covered_subtopics=["Moyenne"])
    planned = _planned(subtopics=["Moyenne", "Variance"])
    issues = _completeness_issues(section, planned)
    assert any("Comment" in i for i in issues)  # sous-section manquante
    assert any("Variance" in i for i in issues)  # sous-thème manquant
    assert any("visuel" in i for i in issues)  # aucun bloc non textuel


def test_completeness_issues_empty_for_fully_conform_section():
    section = _full_section(
        covered_subtopics=["Moyenne"],
        extra_comment_blocks=[{"type": "table", "table": {"headers": ["h"], "rows": [["x"]]}}],
    )
    planned = _planned(subtopics=["Moyenne"])
    assert _completeness_issues(section, planned) == []


def test_completeness_issues_never_flags_an_already_incomplete_fallback_section():
    incomplete = Section.model_validate({
        "type": "development", "title": "S", "blocks": [],
        "subsections": [{"title": "Comment", "blocks": [{"type": "text", "text": INCOMPLETE_SECTION_NOTICE}]}],
    })
    assert is_incomplete_section(incomplete)
    assert _completeness_issues(incomplete, _planned(subtopics=["Moyenne"])) == []


# ─── Lot C : boucle de réparation bornée ─────────────────────────────────────


@pytest.mark.asyncio
async def test_repair_stops_as_soon_as_batch_is_complete():
    bad = _full_section(title="A", subsection_titles=("Pourquoi",))
    good = _full_section(
        title="B", covered_subtopics=["x"],
        extra_comment_blocks=[{"type": "list", "list_items": ["a"]}],
    )
    batch = [_planned("A"), _planned("B", subtopics=["x"])]
    fixed = _full_section(
        title="A", covered_subtopics=[],
        extra_comment_blocks=[{"type": "list", "list_items": ["a"]}],
    )
    client = AsyncMock()
    client.format_structured.return_value = {"sections": [fixed.model_dump(mode="json")]}

    result = await _repair_incomplete_sections(
        [bad, good], batch, question="q", mode="question_only", context_block="c", outline="o", gemini_client=client
    )

    assert client.format_structured.await_count == 1  # complet dès la 1re tentative : pas de 2e appel
    assert _completeness_issues(result[0], batch[0]) == []
    assert result[1] is good  # section déjà conforme jamais retouchée


@pytest.mark.asyncio
async def test_repair_retries_up_to_max_attempts_then_stops():
    bad = _full_section(title="A", subsection_titles=("Pourquoi",))
    batch = [_planned("A")]
    # Chaque tentative renvoie une section toujours incomplète (Comment reste vide) :
    # la boucle doit s'arrêter après _MAX_REPAIR_ATTEMPTS appels, jamais boucler indéfiniment.
    still_bad = _full_section(title="A", subsection_titles=("Pourquoi", "Quoi"))
    client = AsyncMock()
    client.format_structured.return_value = {"sections": [still_bad.model_dump(mode="json")]}

    result = await _repair_incomplete_sections(
        [bad], batch, question="q", mode="m", context_block="c", outline="o", gemini_client=client
    )

    assert client.format_structured.await_count == 2  # _MAX_REPAIR_ATTEMPTS, jamais plus
    assert _structural_gaps(result[0]) == ["Comment"]  # toujours incomplet : géré ensuite par Lot D


@pytest.mark.asyncio
async def test_repair_keeps_original_sections_on_gemini_failure():
    bad = _full_section(title="A", subsection_titles=("Pourquoi",))
    batch = [_planned("A")]
    client = AsyncMock()
    client.format_structured.side_effect = GeminiUnavailableError("indisponible")

    result = await _repair_incomplete_sections(
        [bad], batch, question="q", mode="m", context_block="c", outline="o", gemini_client=client
    )

    assert result == [bad]
    client.format_structured.assert_awaited_once()  # échoue une fois, n'insiste pas


@pytest.mark.asyncio
async def test_repair_rejects_replacement_that_is_not_strictly_better():
    bad = _full_section(title="A", subsection_titles=("Pourquoi", "Quoi"))  # 1 sous-section manquante
    batch = [_planned("A")]
    # Le remplaçant proposé est tout aussi incomplet (juste une autre sous-section manquante) :
    # pas strictement moins de problèmes, donc pas retenu.
    no_better = _full_section(title="A", subsection_titles=("Quoi", "Comment"))
    client = AsyncMock()
    client.format_structured.return_value = {"sections": [no_better.model_dump(mode="json")]}

    result = await _repair_incomplete_sections(
        [bad], batch, question="q", mode="m", context_block="c", outline="o", gemini_client=client
    )

    assert result == [bad]


# ─── Lot D : filet de sécurité sans appel Gemini ─────────────────────────────


def test_finalize_incomplete_leaves_conform_sections_untouched():
    good = _full_section(
        covered_subtopics=["x"], extra_comment_blocks=[{"type": "list", "list_items": ["a"]}],
    )
    planned = _planned(subtopics=["x"])
    assert _finalize_incomplete([good], [planned]) == [good]


def test_finalize_incomplete_replaces_still_broken_section_with_visible_fallback():
    still_broken = _full_section(subsection_titles=("Pourquoi", "Quoi"))  # Comment manquant
    planned = _planned("Titre du plan", subtopics=["x"])

    result = _finalize_incomplete([still_broken], [planned])

    assert is_incomplete_section(result[0])
    assert result[0].title == "Titre du plan"  # le titre du plan est toujours respecté
