"""Graphes générés à part des sections, par un appel Gemini dédié (`app.services.course_graphs`)."""

import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.api.schemas import ApiPlannedSection, CourseGenerationResponse
from app.core.config import Settings
from app.core.exceptions import GeminiUnavailableError
from app.schemas.course_generation import (
    COURSE_GENERATION_GEMINI_SCHEMA,
    CourseGraphsSchema,
    Section,
    SectionsBatchSchema,
)
from app.services.course_generator import _map_sections_to_course_sections
from app.services.course_graphs import attach_graphs
from app.services.course_plan_generator import _incomplete_section, generate_course_from_validated_plan
from app.services.gemini_client import GeminiClient
from tests.test_course_plan_generator import _api_sections, _dev_section, _plan_row, _wrap_up_structured


def _graph(caption: str = "Parcours", **overrides) -> dict:
    base = {
        "layout": "graph", "directed": False, "caption": caption,
        "nodes": [{"id": "A", "label": "A"}, {"id": "B", "label": "B"}],
        "edges": [{"id": "e1", "source": "A", "target": "B", "label": "3"}],
        "steps": [{"caption": "On part de A.", "highlight_nodes": ["A"], "highlight_edges": ["e1"]}],
    }
    return {**base, **overrides}


def _item(section_index: int, **graph_overrides) -> dict:
    return {"section_index": section_index, "graph": _graph(**graph_overrides)}


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
        meta={"title": "Graphes", "subject": "Algorithmique", "generated_at": "2026-10-12T10:00:00Z"},
        sections=_map_sections_to_course_sections(sections), summary="résumé",
    )


def _settings(**overrides) -> Settings:
    return Settings(gemini_api_key="k", course_graphs_enabled=True, **overrides)


def _gemini(structured: dict | Exception) -> AsyncMock:
    gemini = AsyncMock()
    gemini.is_degraded.return_value = False
    if isinstance(structured, Exception):
        gemini.format_structured.side_effect = structured
    else:
        gemini.format_structured.return_value = structured
    return gemini


def _graph_blocks(response: CourseGenerationResponse) -> dict[str, list[str]]:
    """`{titre de section: [sous-section portant un graphe, ...]}`."""
    found: dict[str, list[str]] = {}
    for section in response.sections:
        for sub in section.subsections:
            if any(block.type == "graph" for block in sub.blocks):
                found.setdefault(section.title, []).append(sub.title)
    return found


@pytest.mark.asyncio
async def test_graph_is_appended_to_the_comment_subsection_of_its_section():
    gemini = _gemini({"graphs": [_item(1, caption="Graphe de Gamma")]})

    result = await attach_graphs(_response(), settings=_settings(), gemini_client=gemini)

    # Les index portent sur les sections proposées au modèle : Alpha (0) et Gamma (1) seulement.
    assert _graph_blocks(result) == {"Gamma": ["Comment"]}
    comment = next(sub for sub in result.sections[3].subsections if sub.title == "Comment")
    assert comment.blocks[-1].graph.caption == "Graphe de Gamma"
    assert comment.blocks[-1].graph.steps[0].highlight_edges == ["e1"]
    assert gemini.format_structured.await_args.kwargs["response_schema"] is CourseGraphsSchema


@pytest.mark.asyncio
async def test_only_written_development_sections_are_offered_to_the_model():
    gemini = _gemini({"graphs": []})

    await attach_graphs(_response(), settings=_settings(), gemini_client=gemini)

    prompt = gemini.format_structured.await_args.kwargs["raw_answer"]
    listed = re.findall(r"^(\d+)\. (.+)$", prompt.split("<sections>")[1].split("</sections>")[0], re.M)
    assert listed == [("0", "Alpha"), ("1", "Gamma")]  # ni l'introduction, ni la section de repli
    assert "DONNÉE" in prompt and "n'exécute jamais" in gemini.format_structured.await_args.kwargs["system_instruction"]


@pytest.mark.asyncio
async def test_inconsistent_graph_is_dropped_without_losing_the_others():
    orphan_edge = _item(0, edges=[{"id": "e1", "source": "A", "target": "Z"}])
    gemini = _gemini({"graphs": [orphan_edge, _item(1)]})

    result = await attach_graphs(_response(), settings=_settings(), gemini_client=gemini)

    assert _graph_blocks(result) == {"Gamma": ["Comment"]}


@pytest.mark.asyncio
async def test_out_of_range_and_duplicate_section_indexes_are_ignored():
    gemini = _gemini({"graphs": [_item(-1), _item(7), _item(0, caption="premier"), _item(0, caption="doublon")]})

    result = await attach_graphs(_response(), settings=_settings(), gemini_client=gemini)

    assert _graph_blocks(result) == {"Alpha": ["Comment"]}
    comment = next(sub for sub in result.sections[1].subsections if sub.title == "Comment")
    assert [b.graph.caption for b in comment.blocks if b.type == "graph"] == ["premier"]


@pytest.mark.asyncio
async def test_number_of_graphs_is_capped():
    gemini = _gemini({"graphs": [_item(0), _item(1)]})

    result = await attach_graphs(_response(), settings=_settings(course_graphs_max=1), gemini_client=gemini)

    assert _graph_blocks(result) == {"Alpha": ["Comment"]}
    assert "au plus 1 sections" in gemini.format_structured.await_args.kwargs["raw_answer"]


@pytest.mark.asyncio
@pytest.mark.parametrize("settings", [
    Settings(gemini_api_key="k", course_graphs_enabled=False),
    Settings(gemini_api_key="k", course_graphs_enabled=True, course_graphs_max=0),
])
async def test_disabled_makes_no_gemini_call(settings):
    gemini = _gemini({"graphs": [_item(0)]})

    result = await attach_graphs(_response(), settings=settings, gemini_client=gemini)

    assert _graph_blocks(result) == {}
    gemini.format_structured.assert_not_awaited()


@pytest.mark.asyncio
async def test_degraded_gemini_or_course_without_written_section_makes_no_call():
    degraded = _gemini({"graphs": [_item(0)]})
    degraded.is_degraded.return_value = True
    await attach_graphs(_response(), settings=_settings(), gemini_client=degraded)
    degraded.format_structured.assert_not_awaited()

    empty = _response()
    empty.sections = empty.sections[:1]  # introduction seule
    gemini = _gemini({"graphs": [_item(0)]})
    await attach_graphs(empty, settings=_settings(), gemini_client=gemini)
    gemini.format_structured.assert_not_awaited()

    assert await attach_graphs(_response(), settings=_settings(), gemini_client=None)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", [
    GeminiUnavailableError("indisponible"),
    RuntimeError("inattendu"),
    {"graphs": "pas une liste"},
    {},
])
async def test_failure_never_breaks_the_course(outcome):
    result = await attach_graphs(_response(), settings=_settings(), gemini_client=_gemini(outcome))

    assert _graph_blocks(result) == {}
    assert [s.title for s in result.sections] == ["Introduction", "Alpha", "Beta", "Gamma"]


@pytest.mark.asyncio
async def test_section_that_already_has_a_graph_is_not_offered_again():
    response = await attach_graphs(_response(), settings=_settings(), gemini_client=_gemini({"graphs": [_item(0)]}))
    gemini = _gemini({"graphs": []})

    await attach_graphs(response, settings=_settings(), gemini_client=gemini)

    prompt = gemini.format_structured.await_args.kwargs["raw_answer"]
    assert "0. Gamma" in prompt and "Alpha" not in prompt.split("<sections>")[1]


# ─── Schéma envoyé à Gemini ────────────────────────────────────


@pytest.mark.asyncio
async def test_graphs_schema_is_sent_whole_even_though_sections_hide_the_graph_block():
    """Régression réelle : le retrait du bloc GRAPH des schémas de sections plantait (KeyError) sur
    un schéma de graphes sans blocs de contenu, et l'aurait vidé de son objet."""
    fake = MagicMock()
    fake.models.generate_content.return_value = SimpleNamespace(text='{"graphs": []}')
    client = GeminiClient(Settings(gemini_api_key="k", gemini_max_retries=1), client=fake)

    await client.format_structured("raw", response_schema=CourseGraphsSchema, system_instruction="system")

    sent = fake.models.generate_content.call_args.kwargs["config"].response_json_schema
    assert {"GraphData", "GraphNode", "GraphEdge", "GraphStep", "SectionGraph"} <= set(sent["$defs"])


# ─── Bout en bout : cours généré depuis un plan ────────────────


@pytest.mark.asyncio
async def test_course_from_plan_gets_its_graphs_from_one_extra_call():
    calls: list = []

    async def format_structured(raw_answer, system_instruction, *, response_schema=None):
        calls.append(response_schema)
        if response_schema is SectionsBatchSchema:
            count = int(re.search(r"Génère exactement (\d+)", raw_answer).group(1))
            return {"sections": [_dev_section(f"brut-{i}") for i in range(count)]}
        if response_schema is CourseGraphsSchema:
            return {"graphs": [_item(1)]}
        assert response_schema is COURSE_GENERATION_GEMINI_SCHEMA
        return _wrap_up_structured()

    client = AsyncMock()
    client.is_degraded.return_value = False
    client.format_structured.side_effect = format_structured

    result = await generate_course_from_validated_plan(
        _plan_row(), _api_sections(["Alpha", "Beta"]), client, _settings(course_plan_batch_size=5)
    )

    assert _graph_blocks(result) == {"Beta": ["Comment"]}
    assert calls.count(CourseGraphsSchema) == 1 and calls[-1] is CourseGraphsSchema  # après les sections
