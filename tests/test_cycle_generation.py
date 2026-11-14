"""Format « cycle » dans la génération par lots : schéma et consignes du mode, validation, réparation bornée,
sections sans cycle intactes, aucun appel Gemini ajouté dans le cas nominal."""

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.api.schemas import ApiPlannedSection, CourseGenerationResponse
from app.core.config import Settings
from app.core.exceptions import GeminiUnavailableError
from app.schemas.course_generation import (
    COURSE_GENERATION_GEMINI_SCHEMA,
    CycleSectionsBatchSchema,
    Section,
    SectionsBatchSchema,
)
from app.services.course_depth import get_profile
from app.services.course_generator import is_incomplete_section, is_incomplete_section_dict
from app.services.course_plan_generator import (
    _align_batch_sections,
    _box_gaps,
    _finalize_incomplete,
    _format_section,
    _has_content,
    _missing_subtopics,
    _structural_gaps,
    generate_course_from_validated_plan,
    is_better_replacement,
    section_issues,
)
from tests.cycle_fixtures import cycle_section, old_format_section

STANDARD = get_profile("standard")
META = {"title": "Réseaux", "subject": "Informatique", "language": "fr"}


def _planned(title: str = "Sous-réseaux", subtopics: list[str] | None = None, mastery: str | None = None) -> ApiPlannedSection:
    return ApiPlannedSection(type="development", title=title, objective="o", subtopics=subtopics or [], order=2, mastery=mastery)


def _section(mutate=None, title: str = "Sous-réseaux") -> Section:
    return Section.model_validate(cycle_section(title, mutate))


def _wrap_up() -> dict:
    quiz = {
        "question": "?", "choices": ["a", "b"], "correct_indices": [0],
        "difficulty": "difficile", "explanation": "e", "requires_calculation": False,
    }
    return {
        "mode": "file_question", "format": "focused_answer",
        "meta": {"title": "x", "subject": "x", "language": "fr", "generated_at": "2026-10-02T00:00:00"},
        "sources": [],
        "sections": [{"type": "introduction", "title": "Introduction", "blocks": [{"type": "text", "text": "i"}]}],
        "quiz": [quiz, quiz],
        "confidence": "high",
    }


def _fake_gemini(*batch_answers, fail_batches: bool = False) -> AsyncMock:
    """Gemini simulé : clôture valide ; les lots rendent, dans l'ordre, `batch_answers` (une liste de sections),
    puis la dernière réponse en boucle."""
    answers = list(batch_answers)

    async def format_structured(raw_answer, system_instruction, *, response_schema=None):
        if response_schema in (SectionsBatchSchema, CycleSectionsBatchSchema):
            if fail_batches:
                raise GeminiUnavailableError("indisponible")
            sections = answers.pop(0) if len(answers) > 1 else answers[0]
            return {"sections": sections}
        assert response_schema is COURSE_GENERATION_GEMINI_SCHEMA
        return _wrap_up()

    client = AsyncMock()
    client.format_structured.side_effect = format_structured
    return client


def _plan_row(depth: str | None = "standard") -> SimpleNamespace:
    extra = {"depth": depth} if depth else {}
    return SimpleNamespace(
        id=uuid.uuid4(), question="Q", mode="file_question", filenames=[],
        retrieval_context={"chunks": [], "file_sources": [], "search_query": "q", "web_research": None},
        plan={"meta": META, "planned_sections": [], "coverage_notes": ""}, **extra,
    )


def _plan_sections(*dev_titles: str) -> list[ApiPlannedSection]:
    sections = [ApiPlannedSection(type="introduction", title="Introduction", order=1)]
    sections += [ApiPlannedSection(type="development", title=t, order=i) for i, t in enumerate(dev_titles, start=2)]
    return sections


def _calls(client: AsyncMock, *schemas) -> list:
    return [c for c in client.format_structured.call_args_list if c.kwargs["response_schema"] in schemas]


async def _generate(client: AsyncMock, depth: str | None = "standard", titles: tuple[str, ...] = ("Sous-réseaux",), batch_size: int = 4):
    settings = Settings(gemini_api_key="k", course_plan_batch_size=batch_size)
    return await generate_course_from_validated_plan(_plan_row(depth), _plan_sections(*titles), client, settings)


def _dev(response, index: int = 0):
    return [s for s in response.sections if s.title != "Introduction"][index]


# ─── Mode standard : schéma, consignes, sections ────────────────────────────────────────


@pytest.mark.asyncio
async def test_standard_batches_use_the_cycle_schema_and_the_cycle_rules():
    client = _fake_gemini([cycle_section()])

    response = await _generate(client)

    (call,) = _calls(client, CycleSectionsBatchSchema)
    prompt = call.kwargs["raw_answer"]
    assert "au format cycle (objet `cycle`)" in prompt
    assert "pour le `cycle` de chaque section DEVELOPMENT" in prompt and "3 à 5 (vise 4) mini-étapes" in prompt
    assert "mots de prose par section" not in prompt  # les règles de l'ancien format ne sont pas envoyées
    assert not _calls(client, SectionsBatchSchema)
    assert response.meta.depth == "standard"


@pytest.mark.asyncio
async def test_standard_course_sections_carry_a_cycle_but_not_the_introduction():
    client = _fake_gemini([cycle_section()])

    response = await _generate(client)

    intro, dev = response.sections[0], _dev(response)
    assert intro.cycle is None and intro.title == "Introduction"
    assert dev.cycle is not None and dev.incomplete is False
    assert dev.widget_brief is not None and dev.cycle.challenge.options
    assert [s.support.kind for s in dev.cycle.steps] == ["classify", "chart", "fill_table", "order"]


@pytest.mark.asyncio
async def test_nominal_standard_generation_adds_no_gemini_call():
    one = _fake_gemini([cycle_section()])
    await _generate(one)
    two = _fake_gemini([cycle_section("A")], [cycle_section("B")])
    await _generate(two, titles=("A", "B"), batch_size=1)

    assert one.format_structured.await_count == 2  # clôture + 1 lot : aucun appel de réparation
    assert two.format_structured.await_count == 3  # clôture + 2 lots


@pytest.mark.asyncio
@pytest.mark.parametrize("depth", ["express"])
async def test_other_modes_keep_the_old_format_and_schema(depth):
    client = _fake_gemini([old_format_section("Sous-réseaux")])

    response = await _generate(client, depth=depth)

    (call,) = _calls(client, SectionsBatchSchema, CycleSectionsBatchSchema)
    assert call.kwargs["response_schema"] is SectionsBatchSchema
    assert "au format cycle" not in call.kwargs["raw_answer"] and "mots de prose par section" in call.kwargs["raw_answer"]
    assert _dev(response).cycle is None and _dev(response).incomplete is False and _dev(response).subsections


@pytest.mark.asyncio
async def test_cycle_can_be_switched_on_for_another_mode_through_the_profile(monkeypatch):
    """Étendre le cycle à un mode : un seul réglage de profil (`use_cycle`), rien d'autre à toucher."""
    import dataclasses

    from app.services import course_depth

    monkeypatch.setitem(course_depth.PROFILES, "express", dataclasses.replace(course_depth.PROFILES["express"], use_cycle=True))
    express_cycle = cycle_section(mutate=lambda s: s["cycle"]["steps"].pop())  # express : exactement 3 mini-étapes
    express_cycle["cycle"]["traps"]["mode"] = "cards"
    for entry in express_cycle["cycle"]["traps"]["entries"]:
        entry["claim"] = "Idée fausse"
    express_cycle["cycle"]["vocabulary"] = express_cycle["cycle"]["vocabulary"][:2]
    express_cycle["key_takeaways"] = express_cycle["key_takeaways"][:2]
    client = _fake_gemini([express_cycle])

    response = await _generate(client, depth="express")

    assert _calls(client, CycleSectionsBatchSchema) and "3 mini-étapes" in _calls(client, CycleSectionsBatchSchema)[0].kwargs["raw_answer"]
    assert _dev(response).cycle is not None and _dev(response).incomplete is False


@pytest.mark.asyncio
async def test_old_format_section_in_a_standard_course_is_kept_and_not_incomplete():
    client = _fake_gemini([old_format_section("Sous-réseaux")])

    response = await _generate(client)

    dev = _dev(response)
    assert dev.cycle is None and dev.incomplete is False and dev.subsections
    assert client.format_structured.await_count == 2  # ancien format valide : jamais réparé
    assert is_incomplete_section_dict(dev.model_dump(mode="json")) is False


# ─── Réparation bornée, jamais d'appel en plus pour la forme ─────────────────────────────


def _broken_table(section):
    section["cycle"]["steps"][2]["support"]["table"] = {"headers": ["a", "b", "c"], "rows": []}


@pytest.mark.asyncio
async def test_a_missing_part_of_the_cycle_is_repaired_with_one_extra_call():
    client = _fake_gemini([cycle_section(mutate=_broken_table)], [cycle_section()])

    response = await _generate(client)

    assert client.format_structured.await_count == 3  # clôture + lot + 1 réparation
    repair = _calls(client, CycleSectionsBatchSchema)[1].kwargs["raw_answer"]
    assert "cycle : mini-étape 3 : tableau vide" in repair
    assert "pour le `cycle` de chaque section DEVELOPMENT" in repair and "remplis TOUS les temps du cycle" in repair
    assert "Pourquoi/Quoi/Comment" not in repair
    assert _dev(response).incomplete is False and _dev(response).cycle.steps[2].support.table.rows


@pytest.mark.asyncio
async def test_a_part_that_stays_missing_makes_the_section_incomplete_after_bounded_repairs():
    client = _fake_gemini([cycle_section(mutate=_broken_table)])

    response = await _generate(client)

    assert client.format_structured.await_count == 4  # clôture + lot + 2 réparations (borne)
    dev = _dev(response)
    assert dev.incomplete is True and dev.cycle is None
    assert is_incomplete_section_dict(dev.model_dump(mode="json")) is True


@pytest.mark.asyncio
async def test_form_problems_alone_never_trigger_a_repair_call_and_keep_the_cycle():
    def verbose(section):
        section["cycle"]["steps"][0]["text"] = ("mot " * 90).strip() + "."  # RG-4 : 80 mots
        section["cycle"]["steps"][3]["support"] = dict(section["cycle"]["steps"][2]["support"])  # RG-3 (fill_table x2)

    client = _fake_gemini([cycle_section(mutate=verbose)])

    response = await _generate(client)

    assert client.format_structured.await_count == 2
    assert _dev(response).incomplete is False and len(_dev(response).cycle.steps) == 4


@pytest.mark.asyncio
async def test_all_batches_failing_still_raises_as_before():
    client = _fake_gemini([cycle_section()], fail_batches=True)

    with pytest.raises(Exception, match="tous les lots"):
        await _generate(client)


@pytest.mark.asyncio
async def test_cycle_survives_the_persistence_round_trip():
    """Le cours est enregistré en JSONB (`model_dump`) puis relu : le cycle, ses index mélangés et le brief survivent."""
    response = await _generate(_fake_gemini([cycle_section()]))

    stored = response.model_dump(mode="json")
    reloaded = CourseGenerationResponse.model_validate(stored)

    assert _dev(reloaded).cycle == _dev(response).cycle and _dev(reloaded).widget_brief == _dev(response).widget_brief
    stored_dev = [s for s in stored["sections"] if s["title"] != "Introduction"][0]
    assert is_incomplete_section_dict(stored_dev) is False and stored_dev["cycle"]["challenge"]["correct_indices"]


# ─── section_issues, réparation et finalisation ──────────────────────────────────────────


def test_reference_cycle_section_has_no_issue_at_all():
    issues = section_issues(_section(), _planned(), STANDARD)

    assert not issues and issues.all == [] and issues.score == 0


def test_cycle_form_issues_are_listed_but_never_make_the_section_flagged():
    issues = section_issues(_section(lambda s: s["cycle"]["steps"][0].update(text=("mot " * 90).strip() + ".")), _planned(), STANDARD)

    assert issues.soft == () and issues.blocking == () and issues.form
    assert not issues  # pas de réparation à elle seule
    assert issues.all == list(issues.form) and issues.score == len(issues.form)


def test_prediction_revealed_in_the_text_is_a_form_issue_not_a_blocking_one():
    def reveal(section):
        section["cycle"]["steps"][1]["text"] = "On partage 256 adresses : chaque part compte 64 adresses."

    issues = section_issues(_section(reveal), _planned(), STANDARD)

    assert any("réponse de la prédiction apparaît" in f for f in issues.form) and not issues


def test_cycle_structure_gaps_are_blocking_and_replace_the_title_check():
    section = _section(_broken_table)

    issues = section_issues(section, _planned(), STANDARD)

    assert issues and issues.blocking == ("cycle : mini-étape 3 : tableau vide ou absent (`table` obligatoire pour `table` et `fill_table`)",)
    assert issues.blocking_count == 1 and issues.score >= 10
    # Un cycle n'a pas de sous-sections Pourquoi / Quoi / Comment : ce contrôle-là ne s'applique pas.
    assert section.subsections == [] and not any("sous-section" in b for b in issues.blocking)


def test_old_format_sections_keep_the_title_based_structural_check():
    old = Section.model_validate({"type": "development", "title": "S", "blocks": [], "subsections": [{"title": "Pourquoi", "blocks": [{"type": "text", "text": "x"}]}]})

    assert _structural_gaps(old) == ["Quoi", "Comment"]
    # Mode non cycle : messages d'origine ; lot du format cycle : dit en termes de cycle, jamais « sous-section vide ».
    assert section_issues(old, None, get_profile("express")).blocking == ("sous-section « Quoi » vide, à écrire", "sous-section « Comment » vide, à écrire")
    in_cycle_mode = section_issues(old, None, STANDARD)
    assert len(in_cycle_mode.blocking) == 1 and "objet `cycle` absent" in in_cycle_mode.blocking[0]
    assert in_cycle_mode.blocking_count == 2 and not any("sous-section" in b for b in in_cycle_mode.blocking)


def test_old_format_soft_issues_still_trigger_repair_as_before():
    old = Section.model_validate(old_format_section())  # 2 blocs non textuels sur 4 : conforme au ratio du mode
    text_only = Section.model_validate({
        "type": "development", "title": "S", "blocks": [],
        "subsections": [{"title": n, "blocks": [{"type": "text", "text": n}]} for n in ("Pourquoi", "Quoi", "Comment")],
        "key_takeaways": ["a"], "pitfalls_to_avoid": [{"mistake": "m", "how_to_avoid": "h"}],
    })

    assert not section_issues(old, None, STANDARD)
    flagged = section_issues(text_only, None, STANDARD)
    assert flagged.soft and flagged.form == () and flagged


def test_box_gaps_for_a_cycle_expect_takeaways_and_recall_but_not_old_pitfalls():
    assert _box_gaps(_section()) == []
    assert _box_gaps(_section(lambda s: s.update(key_takeaways=[]))) == ["`key_takeaways`"]
    assert _box_gaps(_section(lambda s: s.update(recall_prompt=None))) == ["`recall_prompt`"]
    old = Section.model_validate({"type": "development", "title": "S", "key_takeaways": ["a"]})
    assert _box_gaps(old) == ["`pitfalls_to_avoid`"]


def test_missing_boxes_alone_never_flag_a_cycle_section_but_are_listed():
    issues = section_issues(_section(lambda s: s.update(recall_prompt=None)), _planned(), STANDARD)

    assert not issues and issues.boxes and "`recall_prompt`" in issues.boxes[0]


def test_finalize_marks_a_blocking_cycle_incomplete_and_keeps_a_form_only_one():
    broken, verbose = _section(_broken_table), _section(lambda s: s["cycle"]["steps"][0].update(text=("mot " * 90).strip() + "."))
    batch = [_planned("Sous-réseaux"), _planned("Sous-réseaux")]

    finalized = _finalize_incomplete([broken, verbose], batch, STANDARD)

    assert is_incomplete_section(finalized[0]) and finalized[0].cycle is None
    assert finalized[1] is verbose


def test_a_better_cycle_replacement_is_kept_only_if_its_score_improves():
    current = section_issues(_section(_broken_table), _planned(), STANDARD)
    good = _section()

    assert is_better_replacement(good, current, section_issues(good, _planned(), STANDARD)) is True
    assert is_better_replacement(good, current, current) is False


# ─── Contenu, alignement, couverture des sous-thèmes, mention « maîtrisée » ─────────────────


def test_a_cycle_section_has_content_and_an_empty_cycle_does_not():
    assert _has_content(_section()) is True
    assert _has_content(Section.model_validate({"type": "development", "title": "t", "cycle": {"steps": []}})) is False


def test_align_batch_sections_keeps_cycle_sections_and_replaces_an_empty_one():
    empty = Section.model_validate({"type": "development", "title": "B", "cycle": {"steps": []}})

    aligned = _align_batch_sections([_section(title="A"), empty], [_planned("A"), _planned("B")])

    assert aligned[0].cycle is not None and aligned[0].title == "A"
    assert is_incomplete_section(aligned[1]) and aligned[1].cycle is None


def test_subtopic_coverage_reads_the_text_of_the_cycle():
    section = _section()
    planned = _planned(subtopics=["Masque de sous-réseau", "Routage dynamique OSPF"])

    assert _missing_subtopics(section, planned) == ["Routage dynamique OSPF"]


def test_declared_coverage_still_takes_priority_for_a_cycle():
    section = _section(lambda s: s.update(covered_subtopics=["Routage dynamique OSPF"]))

    assert _missing_subtopics(section, _planned(subtopics=["Routage dynamique OSPF"])) == []


def test_known_section_gets_a_condensed_instruction_compatible_with_the_cycle():
    known = _planned(mastery="known")

    assert "les cinq temps du cycle" in _format_section(known, detailed=True, cycle=True)
    assert "sans défi ni exemple à trous" in _format_section(known, detailed=True)
    assert "sans défi" not in _format_section(known, detailed=True, cycle=True)


# ─── Anciens cours : jamais marqués incomplets ──────────────────────────────────────────


LEGACY_STORED_SECTION = {
    "id": "0", "title": "Ancien", "quoi": "q", "pourquoi": "p", "comment": "c",
    "subsections": [{"title": "Comment", "blocks": [{"type": "text", "text": "mécanisme"}]}],
}


def test_stored_section_without_cycle_is_not_incomplete():
    assert is_incomplete_section_dict(dict(LEGACY_STORED_SECTION)) is False
    assert is_incomplete_section_dict({**LEGACY_STORED_SECTION, "cycle": None}) is False


def test_stored_cycle_section_is_complete_when_it_has_a_challenge_and_steps():
    complete = {"id": "1", "title": "T", "quoi": "", "pourquoi": "", "comment": "", "cycle": {"challenge": {"question": "q"}, "steps": [{"text": "t"}]}}

    assert is_incomplete_section_dict(complete) is False
    assert is_incomplete_section_dict({**complete, "cycle": {"challenge": {"question": "q"}, "steps": []}}) is True
    assert is_incomplete_section_dict({**complete, "cycle": {"steps": [{"text": "t"}]}}) is True


def test_stored_legacy_section_reads_as_before_without_cycle():
    """Relecture d'un cours enregistré avant le cycle : le JSON se lit comme avant, sans cycle ni brief."""
    from app.api.schemas import CourseSection

    old = CourseSection.model_validate({**LEGACY_STORED_SECTION, "worked_example": {"statement": "", "steps": [], "result": ""}})

    assert old.cycle is None and old.widget_brief is None and old.incomplete is False
