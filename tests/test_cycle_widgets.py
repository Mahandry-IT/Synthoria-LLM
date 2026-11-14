"""Visuel simulable d'une section au format cycle : le MÊME appel widget (un par section), construit depuis le
`widget_brief` et rattaché à la mini-étape désignée ; absent sans `widget_brief` ; filtre HTML inchangé."""

import random
from unittest.mock import AsyncMock

import pytest

from app.api.schemas import ApiCycleStep, ApiPlannedSection, ApiWidget, CourseGenerationResponse
from app.core.config import Settings
from app.core.exceptions import GeminiQuotaExceededError
from app.schemas.course_generation import (
    COURSE_GENERATION_GEMINI_SCHEMA,
    CycleSectionsBatchSchema,
    Section,
    SectionWidgetResult,
)
from app.services.course_generator import _map_sections_to_course_sections
from app.services.course_plan_generator import _incomplete_section, generate_course_from_validated_plan
from app.services.course_widgets import _is_eligible, attach_widgets
from tests.cycle_fixtures import cycle_section, old_format_section
from tests.test_cycle_generation import _plan_row, _plan_sections, _wrap_up
from tests.test_course_widgets import _FORBIDDEN, _HTML, _NO_WIDGET, _gemini, _item, _prompts, _settings


def _drop_brief(section: dict) -> None:
    section["widget_brief"] = None


def _response(*sections: Section) -> CourseGenerationResponse:
    intro = Section.model_validate({
        "type": "introduction", "title": "Introduction", "blocks": [{"type": "text", "text": "intro"}], "subsections": [],
    })
    return CourseGenerationResponse(
        mode="question_only", format="full_course", sources=[], introduction={},
        meta={"title": "Réseaux", "subject": "Informatique", "generated_at": "2026-10-12T10:00:00Z"},
        sections=_map_sections_to_course_sections([intro, *sections], rng=random.Random(0)), summary="résumé",
    )


def _cycle(title: str = "Sous-réseaux", mutate=None) -> Section:
    return Section.model_validate(cycle_section(title, mutate))


def _widget_steps(response: CourseGenerationResponse, title: str) -> list[int]:
    """Numéros (1-based) des mini-étapes qui portent un widget, pour la section `title`."""
    section = next(s for s in response.sections if s.title == title)
    return [n for n, step in enumerate(section.cycle.steps, start=1) if step.widget is not None]


@pytest.mark.asyncio
async def test_widget_is_attached_to_the_step_designated_by_the_brief():
    gemini = _gemini(_item(title="Découpe interactive"))
    response = _response(_cycle())  # widget_brief.step_number = 2

    result = await attach_widgets(response, settings=_settings(), gemini_client=gemini)

    assert _widget_steps(result, "Sous-réseaux") == [2]
    section = result.sections[1]
    assert section.cycle.steps[1].widget.title == "Découpe interactive"
    assert section.cycle.steps[1].widget.html == _HTML
    assert all(step.widget is None for i, step in enumerate(section.cycle.steps) if i != 1)
    # Rien n'est ajouté aux sous-sections : le format cycle n'en a pas.
    assert section.subsections == []


@pytest.mark.asyncio
async def test_widget_is_built_by_the_existing_single_call_from_the_brief_and_the_cycle_text():
    gemini = _gemini(_item())
    response = _response(_cycle())

    await attach_widgets(response, settings=_settings(gemini_chain_widgets=["m1", "m2"]), gemini_client=gemini)

    assert gemini.format_structured.await_count == 1
    (call,) = gemini.format_structured.await_args_list
    assert call.kwargs["response_schema"] is SectionWidgetResult
    assert call.kwargs["chain"] == ["m1", "m2"]
    prompt = call.kwargs["raw_answer"]
    assert "Section à illustrer : Sous-réseaux" in prompt
    assert "Curseur du nombre de parts" in prompt  # le brief
    assert "mini-étape 2" in prompt
    assert "Un réseau /24 compte 256 adresses" in prompt  # le texte des mini-étapes
    assert "## Défi" in prompt and "## Vocabulaire" in prompt
    assert "Chaque part a 64 adresses" not in prompt  # ni les réponses du défi ni la correction de l'épreuve
    assert "256 divisé par 4" not in prompt


@pytest.mark.asyncio
async def test_no_brief_means_no_call_and_no_widget():
    gemini = _gemini(_item())
    response = _response(_cycle(mutate=_drop_brief))

    result = await attach_widgets(response, settings=_settings(), gemini_client=gemini)

    assert gemini.format_structured.await_count == 0
    assert _widget_steps(result, "Sous-réseaux") == []
    assert result.sections[1].cycle is not None and result.sections[1].incomplete is False


def test_eligibility_of_a_cycle_section():
    with_brief = _response(_cycle()).sections[1]
    without_brief = _response(_cycle(mutate=_drop_brief)).sections[1]
    assert _is_eligible(with_brief) is True
    assert _is_eligible(without_brief) is False

    already = _response(_cycle()).sections[1]
    already.cycle.steps[1].widget = ApiWidget(title="t", description="d", html=_HTML)
    assert _is_eligible(already) is False  # un widget par section


def test_brief_pointing_past_the_steps_is_not_eligible():
    section = _response(_cycle()).sections[1]
    section.widget_brief = section.widget_brief.model_copy(update={"step_number": len(section.cycle.steps) + 1})
    assert _is_eligible(section) is False


@pytest.mark.asyncio
async def test_refused_widget_is_rewritten_once_then_the_section_is_left_complete():
    gemini = _gemini(_item(**_FORBIDDEN))  # toujours refusé par le filtre HTML
    response = _response(_cycle())

    result = await attach_widgets(response, settings=_settings(), gemini_client=gemini)

    assert gemini.format_structured.await_count == 2  # une proposition + au plus une réécriture : borne inchangée
    assert "REFUSÉE" in _prompts(gemini)[1]
    assert _widget_steps(result, "Sous-réseaux") == []
    section = result.sections[1]
    assert section.incomplete is False and len(section.cycle.steps) == 4  # le repli de reconstruction suffit
    assert any(step.support.kind == "classify" for step in section.cycle.steps)


@pytest.mark.asyncio
async def test_refused_then_accepted_rewrite_is_attached_to_the_step():
    gemini = _gemini(_item(**_FORBIDDEN), _item(title="Réécrit"))

    result = await attach_widgets(_response(_cycle()), settings=_settings(), gemini_client=gemini)

    assert gemini.format_structured.await_count == 2
    assert result.sections[1].cycle.steps[1].widget.title == "Réécrit"


@pytest.mark.asyncio
async def test_model_may_answer_without_widget_and_the_section_is_untouched():
    gemini = _gemini(_NO_WIDGET)

    result = await attach_widgets(_response(_cycle()), settings=_settings(), gemini_client=gemini)

    assert gemini.format_structured.await_count == 1
    assert _widget_steps(result, "Sous-réseaux") == []


@pytest.mark.asyncio
async def test_html_filter_applies_to_the_attached_widget_and_nothing_unsafe_is_stored():
    """Scripts réseau, iframe et accès au parent : refusés comme pour un bloc `widget` ; jamais attachés."""
    unsafe = [
        "<iframe src='x'></iframe>", "<script>fetch('/x')</script>", "<script>window.parent.postMessage(1,'*')</script>",
    ]
    for html in unsafe:
        gemini = _gemini(_item(html=html))
        result = await attach_widgets(_response(_cycle()), settings=_settings(), gemini_client=gemini)
        assert _widget_steps(result, "Sous-réseaux") == []


@pytest.mark.asyncio
async def test_one_call_per_section_budget_and_bounds_are_unchanged():
    gemini = _gemini(_item(title="A"), _item(title="B"), _item(title="C"))
    response = _response(_cycle("A"), _cycle("B"), _cycle("C"))

    result = await attach_widgets(response, settings=_settings(course_widgets_max=2), gemini_client=gemini)

    assert gemini.format_structured.await_count == 2  # `course_widgets_max` borne les widgets
    assert [_widget_steps(result, t) for t in "ABC"] == [[2], [2], []]

    gemini = _gemini(_item(**_FORBIDDEN))
    fresh = _response(_cycle("A"), _cycle("B"), _cycle("C"))
    await attach_widgets(fresh, settings=_settings(course_widgets_max_calls=3), gemini_client=gemini)
    assert gemini.format_structured.await_count == 3  # `course_widgets_max_calls` aussi, réécritures comprises


@pytest.mark.asyncio
async def test_old_format_sections_still_get_their_widget_in_comment_and_the_mix_costs_one_call_each():
    old = Section.model_validate(old_format_section("Ancien"))
    gemini = _gemini(_item(title="Widget ancien"), _item(title="Widget cycle"))
    response = _response(old, _cycle(), _cycle("Sans visuel", mutate=_drop_brief))

    result = await attach_widgets(response, settings=_settings(), gemini_client=gemini)

    assert gemini.format_structured.await_count == 2  # l'ancienne section + la section du cycle avec brief
    comment = next(sub for sub in result.sections[1].subsections if sub.title == "Comment")
    assert comment.blocks[-1].type == "widget" and comment.blocks[-1].widget.title == "Widget ancien"
    assert result.sections[1].cycle is None
    assert result.sections[2].cycle.steps[1].widget.title == "Widget cycle"
    assert _widget_steps(result, "Sans visuel") == []


@pytest.mark.asyncio
async def test_incomplete_fallback_section_is_never_illustrated():
    fallback = _incomplete_section(ApiPlannedSection(type="development", title="Repli", order=3))
    gemini = _gemini(_item())

    result = await attach_widgets(_response(fallback), settings=_settings(), gemini_client=gemini)

    assert gemini.format_structured.await_count == 0
    assert result.sections[1].incomplete is True


@pytest.mark.asyncio
async def test_quota_error_keeps_the_cycle_section_as_it_is():
    gemini = _gemini(GeminiQuotaExceededError("quota"))

    result = await attach_widgets(_response(_cycle()), settings=_settings(), gemini_client=gemini)

    assert _widget_steps(result, "Sous-réseaux") == []
    assert result.sections[1].cycle is not None


def test_stored_step_widget_round_trips_and_an_unsafe_one_is_dropped_on_read_without_losing_the_cycle():
    step = {
        "text": "t", "support": {"kind": "order", "order_items": ["a", "b"]},
        "widget": {"title": "T", "description": "D", "html": _HTML},
    }
    assert ApiCycleStep.model_validate(step).widget.html == _HTML

    unsafe = {**step, "widget": {"title": "T", "description": "D", "html": "<script>fetch('/x')</script>"}}
    read = ApiCycleStep.model_validate(unsafe)
    assert read.widget is None and read.text == "t"  # le widget est écarté, pas la mini-étape


def test_widget_persistence_round_trip_keeps_it_on_its_step():
    response = _response(_cycle())
    response.sections[1].cycle.steps[1].widget = ApiWidget(title="T", description="D", html=_HTML)

    reloaded = CourseGenerationResponse.model_validate(response.model_dump(mode="json"))

    assert reloaded.sections[1].cycle.steps[1].widget.title == "T"
    assert reloaded.sections[1].cycle.steps[0].widget is None



# ─── Génération complète : nombre d'appels Gemini ────────────────────────────────────────────────────


def _full_flow_gemini(briefs: bool) -> AsyncMock:
    """Gemini simulé d'un cours standard de deux sections : lots, clôture et widgets (client non dégradé)."""
    sections = [cycle_section("A", None if briefs else _drop_brief), cycle_section("B", None if briefs else _drop_brief)]

    async def format_structured(raw_answer, system_instruction, *, response_schema=None, **_):
        if response_schema is CycleSectionsBatchSchema:
            return {"sections": sections}
        if response_schema is SectionWidgetResult:
            return _item(title="Visuel")
        assert response_schema is COURSE_GENERATION_GEMINI_SCHEMA
        return _wrap_up()

    client = AsyncMock()
    client.is_degraded.return_value = False
    client.format_structured.side_effect = format_structured
    return client


async def _run_full_flow(client: AsyncMock) -> CourseGenerationResponse:
    settings = Settings(gemini_api_key="k", course_plan_batch_size=4, course_widgets_enabled=True)
    return await generate_course_from_validated_plan(_plan_row("standard"), _plan_sections("A", "B"), client, settings)


def _calls_by_schema(client: AsyncMock) -> dict[str, int]:
    counts: dict[str, int] = {}
    for call in client.format_structured.await_args_list:
        name = getattr(call.kwargs["response_schema"], "__name__", "closing")
        counts[name] = counts.get(name, 0) + 1
    return counts


@pytest.mark.asyncio
async def test_full_generation_costs_one_widget_call_per_section_with_a_brief_and_nothing_else():
    client = _full_flow_gemini(briefs=True)

    response = await _run_full_flow(client)

    assert _calls_by_schema(client) == {"CycleSectionsBatchSchema": 1, "closing": 1, "SectionWidgetResult": 2}
    dev = [s for s in response.sections if s.title in ("A", "B")]
    assert [[n for n, st in enumerate(s.cycle.steps, 1) if st.widget] for s in dev] == [[2], [2]]
    assert all(s.cycle.minutes_min == 5 for s in dev)  # durée posée en même temps que le mode, sans appel


@pytest.mark.asyncio
async def test_full_generation_without_briefs_makes_no_widget_call_at_all():
    client = _full_flow_gemini(briefs=False)

    response = await _run_full_flow(client)

    assert _calls_by_schema(client) == {"CycleSectionsBatchSchema": 1, "closing": 1}
    assert all(step.widget is None for s in response.sections if s.cycle for step in s.cycle.steps)
    assert all(not s.incomplete for s in response.sections)
