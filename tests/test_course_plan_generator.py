import re
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.api.schemas import ApiPlannedSection
from app.core.config import Settings
from app.core.exceptions import GeminiInvalidResponseError, GeminiUnavailableError
from app.schemas.course_generation import (
    CoursePlanSchema,
    CourseGenerationSchema,
    SectionsBatchSchema,
)
from app.services.course_plan_generator import (
    _align_batch_sections,
    _incomplete_section,
    generate_course_from_validated_plan,
    generate_course_plan,
)

CHUNK = {"content": "extrait", "metadata": {"filename": "doc.pdf", "page": 2}, "distance": 0.1}


def _planned(order: int, type_: str, title: str, subtopics: list[str] | None = None) -> dict:
    return {
        "type": type_,
        "title": title,
        "objective": f"Objectif de {title}",
        "subtopics": subtopics or ["notion a", "notion b"],
        "order": order,
    }


PLAN_STRUCTURED = {
    "meta": {"title": "Transformateurs", "subject": "Électrotechnique", "language": "fr"},
    "planned_sections": [
        # Volontairement désordonné : le service doit trier par `order`.
        _planned(3, "development", "Rendement"),
        _planned(1, "introduction", "Introduction"),
        _planned(2, "development", "Principe"),
        _planned(4, "summary", "Résumé"),
    ],
    "coverage_notes": "RAS",
}


@pytest.fixture
def settings() -> Settings:
    return Settings(gemini_api_key="fake-key", gemini_use_search_grounding=True, course_plan_batch_size=4)


@pytest.fixture
def vector_store():
    store = AsyncMock()
    store.search.return_value = [dict(CHUNK)]
    return store


@pytest.fixture
def gemini_client():
    client = AsyncMock()
    client.reformulate_query.return_value = "requête reformulée"
    client.search_grounded.return_value = ("synthèse web", [{"type": "web", "label": "W", "reference": "https://w"}])
    client.format_structured.return_value = PLAN_STRUCTURED
    return client


# ─── generate_course_plan ────────────────────────────────────


@pytest.mark.asyncio
async def test_generate_plan_sorted_renumbered_without_course_content(settings, vector_store, gemini_client):
    plan, context = await generate_course_plan(
        question="Q", vector_store=vector_store, gemini_client=gemini_client,
        settings=settings, mode="file_question",
    )

    assert [s.title for s in plan.planned_sections] == ["Introduction", "Principe", "Rendement", "Résumé"]
    assert [s.order for s in plan.planned_sections] == [1, 2, 3, 4]
    dumped = plan.model_dump()
    assert "subsections" not in dumped["planned_sections"][0]
    assert context["chunks"] == [{"content": "extrait", "metadata": {"filename": "doc.pdf", "page": 2}}]
    assert context["web_research"] is None
    gemini_client.search_grounded.assert_not_awaited()
    assert gemini_client.format_structured.call_args.kwargs["response_schema"] is CoursePlanSchema


@pytest.mark.asyncio
async def test_generate_plan_question_only_freezes_web_research(settings, vector_store, gemini_client):
    _, context = await generate_course_plan(
        question="Q", vector_store=vector_store, gemini_client=gemini_client,
        settings=settings, mode="question_only",
    )

    gemini_client.search_grounded.assert_awaited_once()
    vector_store.search.assert_not_awaited()
    assert context["web_research"]["raw_answer"] == "synthèse web"
    prompt = gemini_client.format_structured.call_args.kwargs["raw_answer"]
    assert "synthèse web" in prompt


@pytest.mark.asyncio
async def test_generate_plan_question_only_without_grounding_skips_search(vector_store, gemini_client):
    settings = Settings(gemini_api_key="k", gemini_use_search_grounding=False)
    _, context = await generate_course_plan(
        question="Q", vector_store=vector_store, gemini_client=gemini_client,
        settings=settings, mode="question_only",
    )

    gemini_client.search_grounded.assert_not_awaited()
    assert context["web_research"] is None


@pytest.mark.asyncio
async def test_generate_plan_without_development_section_is_invalid(settings, vector_store, gemini_client):
    gemini_client.format_structured.return_value = {
        **PLAN_STRUCTURED,
        "planned_sections": [_planned(1, "introduction", "Introduction")],
    }

    with pytest.raises(GeminiInvalidResponseError):
        await generate_course_plan(
            question="Q", vector_store=vector_store, gemini_client=gemini_client,
            settings=settings, mode="file_question",
        )


# ─── generate_course_from_validated_plan ─────────────────────


def _plan_row() -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        question="Explique les transformateurs",
        mode="file_question",
        filenames=["doc.pdf"],
        retrieval_context={
            "chunks": [{"content": "extrait", "metadata": {"filename": "doc.pdf", "page": 2}}],
            "file_sources": [
                {"type": "file", "label": "doc.pdf", "reference": "page 2"},
                {"type": "file", "label": "doc.pdf", "reference": "page 2"},
            ],
            "search_query": "q",
            "web_research": None,
        },
        plan={"meta": PLAN_STRUCTURED["meta"], "planned_sections": [], "coverage_notes": ""},
    )


def _api_sections(dev_titles: list[str]) -> list[ApiPlannedSection]:
    sections = [_planned(1, "introduction", "Introduction")]
    sections += [_planned(i + 2, "development", t) for i, t in enumerate(dev_titles)]
    n = len(sections)
    sections += [
        _planned(n + 1, "common_pitfalls", "Pièges"),
        _planned(n + 2, "summary", "Résumé"),
        _planned(n + 3, "next_steps", "Suite"),
    ]
    return [ApiPlannedSection(**s) for s in sections]


def _dev_section(title: str) -> dict:
    return {
        "type": "development",
        "title": title,
        "blocks": [],
        "subsections": [
            {"title": "Quoi", "blocks": [{"type": "text", "text": f"quoi {title}"}]},
            {"title": "Pourquoi", "blocks": [{"type": "text", "text": f"pourquoi {title}"}]},
            {"title": "Comment", "blocks": [{"type": "text", "text": f"comment {title}"}]},
        ],
    }


def _wrap_up_structured() -> dict:
    def block(t: str) -> dict:
        return {"type": "text", "text": t}

    def quiz(d: str) -> dict:
        return {
            "question": "?", "choices": ["a", "b"], "correct_indices": [0],
            "difficulty": d, "explanation": "e", "requires_calculation": False,
        }

    return {
        "mode": "question_only",  # halluciné : doit être écrasé par le mode réel
        "format": "full_course",
        "meta": {"title": "x", "subject": "x", "language": "fr", "generated_at": "n'importe quoi"},
        "sources": [],
        "sections": [
            {"type": "introduction", "title": "Introduction", "blocks": [block("intro")], "subsections": []},
            {"type": "common_pitfalls", "title": "Pièges", "blocks": [
                {"type": "pitfall", "pitfall": {"description": "d", "why_it_happens": "w", "how_to_avoid": "h"}}
            ], "subsections": []},
            {"type": "summary", "title": "Résumé", "blocks": [block("résumé")], "subsections": []},
            {"type": "next_steps", "title": "Suite", "blocks": [
                {"type": "list", "list_items": ["a", "b", "c"]}
            ], "subsections": []},
        ],
        "quiz": [quiz("difficile"), quiz("difficile"), quiz("normale"), quiz("facile")],
        "confidence": "high",
        "unconfirmed_points": [],
    }


def _fake_gemini(fail_batches: set[int] | None = None, fail_wrap_up: bool = False) -> AsyncMock:
    """Gemini simulé : un lot renvoie autant de sections que demandé (« Génère exactement N »)."""
    fail_batches = fail_batches or set()
    state = {"batch": 0}

    async def format_structured(raw_answer, system_instruction, *, response_schema=None):
        if response_schema is SectionsBatchSchema:
            state["batch"] += 1
            if state["batch"] in fail_batches:
                raise GeminiUnavailableError("indisponible")
            count = int(re.search(r"Génère exactement (\d+)", raw_answer).group(1))
            return {"sections": [_dev_section(f"brut-{state['batch']}-{i}") for i in range(count)]}
        assert response_schema is CourseGenerationSchema
        if fail_wrap_up:
            raise GeminiUnavailableError("indisponible")
        return _wrap_up_structured()

    client = AsyncMock()
    client.format_structured.side_effect = format_structured
    return client


def _batch_calls(client: AsyncMock) -> list:
    return [c for c in client.format_structured.call_args_list if c.kwargs["response_schema"] is SectionsBatchSchema]


@pytest.mark.asyncio
async def test_from_plan_batches_and_keeps_planned_titles(settings):
    titles = [f"Notion {i}" for i in range(10)]
    client = _fake_gemini()

    result = await generate_course_from_validated_plan(_plan_row(), _api_sections(titles), client, settings)

    assert len(_batch_calls(client)) == 3  # 10 sections / lots de 4
    assert [s.title for s in result.sections] == ["Introduction"] + titles
    assert result.mode == "file_question"
    assert result.format == "focused_answer"
    assert result.meta.title == "Transformateurs"
    assert len(result.quiz) == 4
    assert result.common_pitfalls is not None
    assert result.next_steps == ["a", "b", "c"]
    # Sources dérivées du contexte figé et dédupliquées
    assert [(s.label, s.reference) for s in result.sources] == [("doc.pdf", "page 2")]


@pytest.mark.asyncio
async def test_from_plan_no_silent_truncation_on_25_plus_sections(settings):
    titles = [f"Notion {i}" for i in range(27)]

    result = await generate_course_from_validated_plan(_plan_row(), _api_sections(titles), _fake_gemini(), settings)

    assert [s.title for s in result.sections] == ["Introduction"] + titles


@pytest.mark.asyncio
async def test_from_plan_reflects_user_edits(settings):
    # L'utilisateur a renommé, supprimé et réordonné des sections.
    sections = _api_sections(["Alpha", "Beta", "Gamma"])
    edited = [s for s in sections if s.title != "Beta"]
    edited = [s.model_copy(update={"title": "Gamma renommée"}) if s.title == "Gamma" else s for s in edited]
    for s in edited:
        if s.title == "Alpha":
            s.order = 50  # passe après Gamma

    client = _fake_gemini()
    result = await generate_course_from_validated_plan(_plan_row(), edited, client, settings)

    assert [s.title for s in result.sections] == ["Introduction", "Gamma renommée", "Alpha"]
    assert "Beta" not in client.format_structured.call_args_list[0].kwargs["raw_answer"].split("CE lot")[1]


@pytest.mark.asyncio
async def test_from_plan_failed_batch_is_marked_incomplete_not_fatal(settings):
    titles = [f"Notion {i}" for i in range(8)]  # 2 lots
    client = _fake_gemini(fail_batches={2})

    result = await generate_course_from_validated_plan(_plan_row(), _api_sections(titles), client, settings)

    by_title = {s.title: s for s in result.sections}
    assert "incomplète" in by_title["Notion 5"].comment
    assert "incomplète" not in by_title["Notion 1"].comment
    assert len(result.sections) == 9


@pytest.mark.asyncio
async def test_from_plan_all_batches_failing_raises(settings):
    client = _fake_gemini(fail_batches={1, 2})

    with pytest.raises(GeminiInvalidResponseError):
        await generate_course_from_validated_plan(
            _plan_row(), _api_sections([f"N{i}" for i in range(8)]), client, settings
        )


@pytest.mark.asyncio
async def test_from_plan_wrap_up_failure_degrades_gracefully(settings):
    result = await generate_course_from_validated_plan(
        _plan_row(), _api_sections(["Alpha", "Beta"]), _fake_gemini(fail_wrap_up=True), settings
    )

    assert [s.title for s in result.sections] == ["Alpha", "Beta"]
    assert result.quiz is None
    assert "relancez" in result.next_steps[0]


def test_align_batch_sections_pads_missing_with_incomplete_placeholder():
    planned = _api_sections(["A", "B"])[1:3]
    returned = SectionsBatchSchema.model_validate({"sections": [_dev_section("A")]}).sections

    aligned = _align_batch_sections(returned, planned)

    assert [s.title for s in aligned] == ["A", "B"]
    assert aligned[1] == _incomplete_section(planned[1])
