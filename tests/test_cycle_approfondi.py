"""Mode approfondi au format « cycle » : schéma, règles, mini-étapes 5 à 6, pièges `spot_rewrite`, durée 8 à 10 min.
Express reste à l'ancien format. Aucun appel Gemini réel (client mocké)."""

import random
import pytest

from app.core.config import Settings
from app.schemas.course_generation import CycleSectionsBatchSchema, Section, SectionsBatchSchema
from app.services.course_depth import PROFILES, estimate_minutes, get_profile, render_cycle_rules
from app.services.course_generator import _map_sections_to_course_sections
from app.services.course_section_adder import add_course_sections
from app.services.cycle_validation import cycle_bounds_issues, cycle_structural_gaps
from app.services.visual_validation import cycle_form_issues
from app.services.section_regenerator import regenerate_section
from tests.cycle_fixtures import approfondi_cycle_section, old_format_section
from tests.test_cycle_generation import _calls, _dev, _fake_gemini, _generate
from tests.test_cycle_regeneration import _gemini, _prompt, _row, _schema, _store

APPROFONDI = get_profile("approfondi")


def test_approfondi_profile_matches_the_spec():
    c = APPROFONDI.cycle
    assert APPROFONDI.use_cycle and not PROFILES["express"].use_cycle
    assert (c.steps_min, c.steps_max, c.steps_target) == (5, 6, 5)
    assert (c.predictions_min, c.predictions_max) == (2, 2) and c.traps_mode == "spot_rewrite"
    assert (c.minutes_min, c.minutes_max) == (8, 10)


@pytest.mark.parametrize(("steps", "expected"), [(5, (8, 10)), (6, (9, 11))])
def test_approfondi_duration(steps, expected):
    assert estimate_minutes("approfondi", steps) == expected


@pytest.mark.parametrize(("steps", "ok"), [(4, False), (5, True), (6, True), (7, False)])
def test_approfondi_accepts_five_or_six_steps(steps, ok):
    raw = approfondi_cycle_section(steps=6)
    raw["cycle"]["steps"] = (raw["cycle"]["steps"] + raw["cycle"]["steps"][2:3])[:steps]  # 7e : fill_table
    section = Section.model_validate(raw)

    gaps = cycle_structural_gaps(section, APPROFONDI)

    assert any("mini-étape(s) pour 5 à 6" in g for g in gaps) is (not ok)
    if ok:
        assert gaps == [] and cycle_form_issues(section, APPROFONDI) == [] and cycle_bounds_issues(section) == []


def test_spot_rewrite_traps_and_two_predictions_are_required_in_approfondi():
    section = Section.model_validate(approfondi_cycle_section())
    assert cycle_form_issues(section, APPROFONDI) == []

    plain_spot = approfondi_cycle_section()
    plain_spot["cycle"]["traps"]["mode"] = "spot"
    assert any("spot_rewrite" in i for i in cycle_form_issues(Section.model_validate(plain_spot), APPROFONDI))

    one_prediction = approfondi_cycle_section()
    one_prediction["cycle"]["steps"][3]["prediction"] = None
    assert cycle_form_issues(Section.model_validate(one_prediction), APPROFONDI)


def test_mapped_approfondi_section_carries_its_duration():
    (section,) = _map_sections_to_course_sections(
        [Section.model_validate(approfondi_cycle_section(steps=6))], rng=random.Random(0), depth="approfondi"
    )
    assert (section.cycle.minutes_min, section.cycle.minutes_max) == (9, 11)
    assert section.cycle.traps.mode == "spot_rewrite" and len(section.cycle.steps) == 6


@pytest.mark.asyncio
@pytest.mark.parametrize("depth", ["approfondi", None])  # None : plan sans mode = approfondi
async def test_approfondi_generation_uses_the_cycle_schema_and_rules(depth):
    client = _fake_gemini([approfondi_cycle_section()])

    response = await _generate(client, depth=depth)

    (call,) = _calls(client, CycleSectionsBatchSchema)
    prompt = call.kwargs["raw_answer"]
    assert render_cycle_rules(APPROFONDI) in prompt and "5 à 6 (vise 5) mini-étapes" in prompt
    assert "mots de prose par section" not in prompt and not _calls(client, SectionsBatchSchema)
    dev = _dev(response)
    assert dev.cycle is not None and dev.incomplete is False
    assert client.format_structured.await_count == 2  # clôture + lot : aucune réparation
    assert response.meta.depth == "approfondi"


@pytest.mark.asyncio
async def test_express_generation_stays_in_the_old_format():
    client = _fake_gemini([old_format_section("Sous-réseaux")])

    response = await _generate(client, depth="express")

    (call,) = _calls(client, SectionsBatchSchema)
    assert call.kwargs["response_schema"] is SectionsBatchSchema and not _calls(client, CycleSectionsBatchSchema)
    assert _dev(response).cycle is None


@pytest.mark.asyncio
@pytest.mark.parametrize("depth", ["approfondi", None])
async def test_approfondi_regeneration_and_addition_use_the_cycle(depth):
    gemini = _gemini([approfondi_cycle_section("Notion")])
    result = await regenerate_section(_row(depth), "Notion", gemini, _store(), Settings(gemini_api_key="k"))

    assert _schema(gemini) is CycleSectionsBatchSchema and render_cycle_rules(APPROFONDI) in _prompt(gemini)
    assert result.cycle is not None

    gemini = _gemini([approfondi_cycle_section("Nouvelle")])
    sections, _ = await add_course_sections(_row(depth), "Sujet", gemini, _store(), Settings(gemini_api_key="k"))
    assert _schema(gemini) is CycleSectionsBatchSchema and "`cycle` complet" in _prompt(gemini)
    assert sections[0].cycle is not None
