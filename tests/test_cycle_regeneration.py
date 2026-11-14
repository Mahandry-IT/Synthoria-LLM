"""Régénération et ajout de section au format cycle (mode standard), prompt de réparation, anciens modes inchangés."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.core.exceptions import GeminiInvalidResponseError
from app.schemas.course_generation import CycleSectionsBatchSchema, SectionsBatchSchema
from app.services.course_depth import get_profile, render_cycle_rules
from app.services.course_section_adder import add_course_sections
from app.services.section_regenerator import regenerate_section
from tests.cycle_fixtures import cycle_section, old_format_section


def _row(depth: str | None, sections: list[dict] | None = None, next_steps: list[str] | None = None) -> SimpleNamespace:
    meta = {"title": "Réseaux", "subject": "Informatique"} | ({"depth": depth} if depth else {})
    return SimpleNamespace(
        question="Comment découper un réseau ?", mode="file_question", filenames=[],
        gemini_response={"meta": meta, "sections": sections or [], "next_steps": next_steps or []},
    )


def _gemini(*answers: list[dict]) -> AsyncMock:
    gemini = AsyncMock()
    gemini.reformulate_query.return_value = "réseau"
    gemini.format_structured.side_effect = [{"sections": sections} for sections in answers]
    return gemini


def _store() -> AsyncMock:
    store = AsyncMock()
    store.search.return_value = []
    return store


def _prompt(gemini: AsyncMock, call: int = -1) -> str:
    return gemini.format_structured.await_args_list[call].kwargs["raw_answer"]


def _schema(gemini: AsyncMock, call: int = -1):
    return gemini.format_structured.await_args_list[call].kwargs["response_schema"]


def _too_few_steps(section: dict) -> None:
    section["cycle"]["steps"] = section["cycle"]["steps"][:2]  # standard : 3 à 5 mini-étapes


def _no_reconstruction(section: dict) -> None:
    section["widget_brief"] = None
    for step in section["cycle"]["steps"]:
        if step["support"]["kind"] in ("classify", "fill_table", "order"):
            step["support"] = {"kind": "table", "caption": "c", "table": {"headers": ["a", "b", "c"], "rows": [["1", "2", "3"]]}}


# ─── Régénération ────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_standard_regeneration_asks_for_a_cycle_section_with_its_schema_and_rules():
    gemini = _gemini([cycle_section("Notion")])

    result = await regenerate_section(_row("standard"), "Notion", gemini, _store(), Settings(gemini_api_key="k"))

    assert gemini.format_structured.await_count == 1  # section conforme : aucune réparation
    assert _schema(gemini) is CycleSectionsBatchSchema
    prompt = _prompt(gemini)
    assert render_cycle_rules(get_profile("standard")) in prompt
    assert "`cycle` complet" in prompt and "`widget_brief`" in prompt
    assert "Pourquoi/Quoi/Comment" not in prompt and "un exemple à trous" not in prompt  # ni la demande de l'ancien format
    assert result.cycle is not None and result.title == "Notion"
    assert result.widget_brief is not None


@pytest.mark.asyncio
async def test_missing_cycle_parts_are_asked_again_in_the_repair_prompt():
    gemini = _gemini([cycle_section("Notion", _too_few_steps)], [cycle_section("Notion")])

    result = await regenerate_section(_row("standard"), "Notion", gemini, _store(), Settings(gemini_api_key="k"))

    assert gemini.format_structured.await_count == 2  # 1 génération + 1 réparation (borne de la régénération)
    assert _schema(gemini, 1) is CycleSectionsBatchSchema
    repair = _prompt(gemini, 1)
    assert "cycle : Sous le capot : 2 mini-étape(s) pour 3 à 5 attendues" in repair  # l'élément manquant est nommé
    assert "remplis TOUS les temps du cycle" in repair and "`widget_brief`" in repair
    assert "Pourquoi, Quoi et Comment" not in repair
    assert len(result.cycle.steps) == 4


@pytest.mark.asyncio
async def test_missing_reconstruction_without_widget_brief_is_repaired_too():
    gemini = _gemini([cycle_section("Notion", _no_reconstruction)], [cycle_section("Notion")])

    result = await regenerate_section(_row("standard"), "Notion", gemini, _store(), Settings(gemini_api_key="k"))

    assert "aucune mini-étape de reconstruction" in _prompt(gemini, 1)
    assert result.widget_brief is not None


@pytest.mark.asyncio
async def test_a_section_still_incomplete_after_repair_is_never_returned():
    broken = cycle_section("Notion", _too_few_steps)
    gemini = _gemini([broken], [broken], [broken])

    with pytest.raises(GeminiInvalidResponseError, match="encore incomplète"):
        await regenerate_section(_row("standard"), "Notion", gemini, _store(), Settings(gemini_api_key="k"))

    assert gemini.format_structured.await_count == 2


@pytest.mark.asyncio
async def test_form_problems_alone_cost_no_repair_call():
    def too_long(section: dict) -> None:
        section["cycle"]["steps"][0]["text"] = " ".join(["mot"] * 120)  # RG-4 : soft, jamais de réparation seule

    gemini = _gemini([cycle_section("Notion", too_long)])

    result = await regenerate_section(_row("standard"), "Notion", gemini, _store(), Settings(gemini_api_key="k"))

    assert gemini.format_structured.await_count == 1
    assert result.cycle is not None


@pytest.mark.asyncio
async def test_invalid_cycle_json_raises_invalid_response():
    bad = cycle_section("Notion")
    bad["cycle"]["steps"] = "pas une liste"
    gemini = _gemini([bad])

    with pytest.raises(GeminiInvalidResponseError):
        await regenerate_section(_row("standard"), "Notion", gemini, _store(), Settings(gemini_api_key="k"))


@pytest.mark.asyncio
@pytest.mark.parametrize("depth", ["express"])
async def test_other_modes_and_old_courses_regenerate_in_the_old_format(depth):
    old = old_format_section("Notion")
    gemini = _gemini([old])

    result = await regenerate_section(_row(depth), "Notion", gemini, _store(), Settings(gemini_api_key="k"))

    assert _schema(gemini) is SectionsBatchSchema
    prompt = _prompt(gemini)
    assert "Pourquoi/Quoi/Comment" in prompt and "`cycle` complet" not in prompt
    assert "pour le `cycle` de chaque section" not in prompt
    assert result.cycle is None


# ─── Ajout de section ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_standard_course_adds_sections_with_a_cycle():
    gemini = _gemini([cycle_section("Le masque"), cycle_section("Les plages")])
    row = _row("standard", sections=[{"title": "Sous-réseaux"}])

    sections, next_steps = await add_course_sections(row, "Parle du masque", gemini, _store(), Settings(gemini_api_key="k"))

    assert [s.title for s in sections] == ["Le masque", "Les plages"]
    assert all(s.cycle is not None and s.widget_brief is not None for s in sections)
    assert _schema(gemini) is CycleSectionsBatchSchema
    prompt = _prompt(gemini)
    assert render_cycle_rules(get_profile("standard")) in prompt
    assert "`cycle` complet" in prompt and "Parle du masque" in prompt
    assert "Pourquoi/Quoi/Comment" not in prompt
    assert "- Sous-réseaux" in prompt  # les sections présentes ne sont pas répétées
    assert gemini.format_structured.await_count == 1  # aucun appel de plus


@pytest.mark.asyncio
@pytest.mark.parametrize("depth", ["express"])
async def test_other_modes_add_old_format_sections(depth):
    gemini = _gemini([old_format_section("Nouvelle")])

    sections, _ = await add_course_sections(_row(depth), "Sujet", gemini, _store(), Settings(gemini_api_key="k"))

    assert _schema(gemini) is SectionsBatchSchema
    assert "Pourquoi/Quoi/Comment" in _prompt(gemini) and "`cycle` complet" not in _prompt(gemini)
    assert sections[0].cycle is None
