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
    MoreSectionsSchema,
    PlannedSection,
    SectionsBatchSchema,
)
from app.services.course_plan_generator import (
    _align_batch_sections,
    _incomplete_section,
    generate_course_from_validated_plan,
    generate_course_plan,
    generate_more_sections,
    refine_planned_section,
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
            {
                "title": "Quoi",
                "blocks": [
                    {"type": "text", "text": f"quoi {title}"},
                    {"type": "table", "table": {"caption": "c", "headers": ["h"], "rows": [["x"]]}},
                ],
            },
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


@pytest.mark.asyncio
async def test_from_plan_regenerates_section_with_uncovered_subtopics(settings):
    sections = [
        ApiPlannedSection(**_planned(1, "introduction", "Introduction")),
        ApiPlannedSection(**_planned(2, "development", "Normes", ["ISO 14001", "ISO 50001"])),
        ApiPlannedSection(**_planned(3, "common_pitfalls", "Pièges")),
        ApiPlannedSection(**_planned(4, "summary", "Résumé")),
        ApiPlannedSection(**_planned(5, "next_steps", "Suite")),
    ]
    base = _fake_gemini()
    batch_prompts: list[str] = []

    async def format_structured(raw_answer, system_instruction, *, response_schema=None):
        if response_schema is not SectionsBatchSchema:
            return await base.format_structured(
                raw_answer=raw_answer, system_instruction=system_instruction, response_schema=response_schema
            )
        batch_prompts.append(raw_answer)
        if len(batch_prompts) == 1:  # 1er jet : n'explique que ISO 14001
            return {"sections": [{**_dev_section("Normes"), "covered_subtopics": ["ISO 14001"]}]}
        return {"sections": [{**_dev_section("Normes v2"), "covered_subtopics": ["ISO 14001", "ISO 50001"]}]}

    client = AsyncMock()
    client.format_structured.side_effect = format_structured

    result = await generate_course_from_validated_plan(_plan_row(), sections, client, settings)

    assert len(batch_prompts) == 2
    assert "ISO 50001" in batch_prompts[1].split("RÉGÉNÉRER")[1]
    normes = next(s for s in result.sections if s.title == "Normes")
    assert normes.quoi == "quoi Normes v2"  # le remplaçant complet est retenu, sous le titre du plan


@pytest.mark.asyncio
async def test_from_plan_keeps_original_when_regeneration_fails(settings):
    sections = [
        ApiPlannedSection(**_planned(1, "introduction", "Introduction")),
        ApiPlannedSection(**_planned(2, "development", "Normes", ["ISO 14001", "ISO 50001"])),
        ApiPlannedSection(**_planned(3, "common_pitfalls", "Pièges")),
        ApiPlannedSection(**_planned(4, "summary", "Résumé")),
        ApiPlannedSection(**_planned(5, "next_steps", "Suite")),
    ]
    base = _fake_gemini()
    calls = {"batch": 0}

    async def format_structured(raw_answer, system_instruction, *, response_schema=None):
        if response_schema is not SectionsBatchSchema:
            return await base.format_structured(
                raw_answer=raw_answer, system_instruction=system_instruction, response_schema=response_schema
            )
        calls["batch"] += 1
        if calls["batch"] == 2:
            raise GeminiUnavailableError("indisponible")
        return {"sections": [{**_dev_section("Normes"), "covered_subtopics": ["ISO 14001"]}]}

    client = AsyncMock()
    client.format_structured.side_effect = format_structured

    result = await generate_course_from_validated_plan(_plan_row(), sections, client, settings)

    normes = next(s for s in result.sections if s.title == "Normes")
    assert normes.quoi == "quoi Normes"


def test_align_batch_sections_pads_missing_with_incomplete_placeholder():
    planned = _api_sections(["A", "B"])[1:3]
    returned = SectionsBatchSchema.model_validate({"sections": [_dev_section("A")]}).sections

    aligned = _align_batch_sections(returned, planned)

    assert [s.title for s in aligned] == ["A", "B"]
    assert aligned[1] == _incomplete_section(planned[1])


# ─── refine_planned_section / generate_more_sections ─────────


def _refine_gemini(client, **overrides):
    client.format_structured.return_value = {
        "type": "development",
        "title": "Principe",
        "objective": "Objectif enrichi",
        "subtopics": ["notion a", "notion b", "notion manquante"],
        "order": 99,
        **overrides,
    }


@pytest.mark.asyncio
async def test_refine_section_keeps_type_and_order_and_adds_missing(settings, vector_store, gemini_client):
    section = ApiPlannedSection(**_planned(3, "development", "Principe"))
    _refine_gemini(gemini_client, type="summary")

    refined = await refine_planned_section(
        _plan_row(), section, _api_sections(["Principe", "Rendement"]), None, vector_store, gemini_client, settings
    )

    assert (refined.type, refined.order) == ("development", 3)
    assert refined.subtopics == ["notion a", "notion b", "notion manquante"]
    assert gemini_client.format_structured.call_args.kwargs["response_schema"] is PlannedSection
    prompt = gemini_client.format_structured.call_args.kwargs["raw_answer"]
    assert "n'a rien précisé" in prompt
    vector_store.search.assert_awaited()


@pytest.mark.asyncio
async def test_refine_section_includes_user_instructions_in_prompt(settings, vector_store, gemini_client):
    section = ApiPlannedSection(**_planned(2, "development", "Principe"))
    _refine_gemini(gemini_client)

    await refine_planned_section(
        _plan_row(), section, [section], "ajoute le cas du transformateur triphasé", vector_store, gemini_client, settings
    )

    prompt = gemini_client.format_structured.call_args.kwargs["raw_answer"]
    assert "transformateur triphasé" in prompt
    assert "n'a rien précisé" not in prompt


@pytest.mark.asyncio
async def test_refine_section_question_only_uses_targeted_web_search(settings, vector_store, gemini_client):
    row = _plan_row()
    row.mode = "question_only"
    section = ApiPlannedSection(**_planned(2, "development", "Principe"))
    _refine_gemini(gemini_client)

    await refine_planned_section(row, section, [section], None, vector_store, gemini_client, settings)

    gemini_client.search_grounded.assert_awaited_once()
    vector_store.search.assert_not_awaited()
    assert "synthèse web" in gemini_client.format_structured.call_args.kwargs["raw_answer"]


@pytest.mark.asyncio
async def test_refine_section_search_failure_is_not_fatal(settings, vector_store, gemini_client):
    vector_store.search.side_effect = GeminiUnavailableError("embedding indisponible")
    section = ApiPlannedSection(**_planned(2, "development", "Principe"))
    _refine_gemini(gemini_client)

    refined = await refine_planned_section(_plan_row(), section, [section], None, vector_store, gemini_client, settings)

    assert refined.objective == "Objectif enrichi"


@pytest.mark.asyncio
async def test_refine_section_clips_to_contract_limits(settings, vector_store, gemini_client):
    section = ApiPlannedSection(**_planned(2, "development", "Principe"))
    _refine_gemini(
        gemini_client, title="T" * 500, objective="O" * 2000, subtopics=["s" * 400] + [f"n{i}" for i in range(40)]
    )

    refined = await refine_planned_section(_plan_row(), section, [section], None, vector_store, gemini_client, settings)

    assert len(refined.title) == 200
    assert len(refined.objective) == 1000
    assert len(refined.subtopics) == 20
    assert len(refined.subtopics[0]) == 300


@pytest.mark.asyncio
async def test_refine_section_empty_subtopics_fall_back_to_previous(settings, vector_store, gemini_client):
    section = ApiPlannedSection(**_planned(2, "development", "Principe", ["ancien"]))
    _refine_gemini(gemini_client, subtopics=[])

    refined = await refine_planned_section(_plan_row(), section, [section], None, vector_store, gemini_client, settings)

    assert refined.subtopics == ["ancien"]


@pytest.mark.asyncio
async def test_refine_section_invalid_response_raises(settings, vector_store, gemini_client):
    section = ApiPlannedSection(**_planned(2, "development", "Principe"))
    gemini_client.format_structured.return_value = {"titre": "mauvais schéma"}

    with pytest.raises(GeminiInvalidResponseError):
        await refine_planned_section(_plan_row(), section, [section], None, vector_store, gemini_client, settings)


@pytest.mark.asyncio
async def test_more_sections_are_development_new_and_ordered_after_plan(gemini_client):
    current = _api_sections(["Principe", "Rendement"])
    gemini_client.format_structured.return_value = {
        "planned_sections": [
            _planned(2, "summary", "Optimisation avancée"),
            _planned(1, "development", "principe"),  # doublon d'un titre existant : ignoré
            _planned(3, "development", "Applications industrielles"),
        ]
    }

    created = (await generate_more_sections(_plan_row(), current, gemini_client)).sections

    assert [s.title for s in created] == ["Optimisation avancée", "Applications industrielles"]
    assert {s.type for s in created} == {"development"}
    assert [s.order for s in created] == [len(current) + 1, len(current) + 2]
    assert gemini_client.format_structured.call_args_list[0].kwargs["response_schema"] is MoreSectionsSchema
    assert "Suite" in gemini_client.format_structured.call_args_list[0].kwargs["raw_answer"]


@pytest.mark.asyncio
async def test_more_sections_capped(gemini_client):
    gemini_client.format_structured.return_value = {
        "planned_sections": [_planned(i, "development", f"Nouveau {i}") for i in range(1, 15)]
    }

    created = (await generate_more_sections(_plan_row(), _api_sections(["A"]), gemini_client)).sections

    assert len(created) == 6


def _plan_with_next_steps() -> list[ApiPlannedSection]:
    return [
        ApiPlannedSection(**_planned(1, "introduction", "Introduction")),
        ApiPlannedSection(**_planned(2, "development", "Principe")),
        ApiPlannedSection(
            **{**_planned(3, "next_steps", "Mes pistes perso", ["Piste A", "Piste B"]), "objective": "Ancien objectif"}
        ),
    ]


@pytest.mark.asyncio
async def test_more_sections_refresh_next_steps_with_new_leads_only(gemini_client):
    current = _plan_with_next_steps()
    gemini_client.format_structured.return_value = {
        "planned_sections": [_planned(4, "development", "Applications industrielles")],
        "next_steps": {
            **_planned(9, "next_steps", "Titre proposé par le modèle"),
            "objective": "Nouvel objectif",
            "subtopics": ["piste a", "Applications industrielles", "Nouvelle piste 1", "Nouvelle piste 2"],
        },
    }

    result = await generate_more_sections(_plan_row(), current, gemini_client)

    assert [s.title for s in result.sections] == ["Applications industrielles"]
    refreshed = result.next_steps
    assert refreshed is not None
    assert refreshed.type == "next_steps"
    assert refreshed.title == "Mes pistes perso"      # titre de l'utilisateur conservé
    assert refreshed.order == 3                        # même position que l'ancienne section
    assert refreshed.objective == "Nouvel objectif"
    # Ni les anciennes pistes (casse ignorée), ni un titre de section déjà présent ou créé
    assert refreshed.subtopics == ["Nouvelle piste 1", "Nouvelle piste 2"]
    prompt = gemini_client.format_structured.call_args.kwargs["raw_answer"]
    assert "next_steps" in prompt and "NOUVELLES pistes" in prompt


@pytest.mark.asyncio
async def test_more_sections_next_steps_none_without_existing_section(gemini_client):
    gemini_client.format_structured.return_value = {
        "planned_sections": [_planned(3, "development", "Nouveau")],
        "next_steps": _planned(9, "next_steps", "Suite", ["Autre piste"]),
    }

    result = await generate_more_sections(_plan_row(), _api_sections(["A"])[:2], gemini_client)

    assert [s.title for s in result.sections] == ["Nouveau"] and result.next_steps is None


@pytest.mark.asyncio
async def test_more_sections_retries_when_model_repeats_same_leads(gemini_client):
    """Pistes identiques (accents/casse/ponctuation ignorés) → relance ciblée, jamais l'ancienne section."""
    gemini_client.format_structured.side_effect = [
        {
            "planned_sections": [_planned(4, "development", "Nouveau")],
            "next_steps": _planned(9, "next_steps", "Suite", ["PISTE  a.", "Piste B !"]),
        },
        {"next_steps": _planned(9, "next_steps", "Suite", ["Piste inédite 1", "Piste inédite 2"])},
    ]

    result = await generate_more_sections(_plan_row(), _plan_with_next_steps(), gemini_client)

    assert gemini_client.format_structured.await_count == 2
    assert result.next_steps is not None
    assert result.next_steps.subtopics == ["Piste inédite 1", "Piste inédite 2"]
    assert result.next_steps.title == "Mes pistes perso" and result.next_steps.order == 3


@pytest.mark.asyncio
async def test_more_sections_retry_then_fallback_drops_developed_leads(gemini_client):
    current = _plan_with_next_steps()
    current[2] = current[2].model_copy(update={"subtopics": ["Nouveau", "Piste B"]})
    same = {"next_steps": _planned(9, "next_steps", "Suite", ["Nouveau", "Piste B"])}
    gemini_client.format_structured.side_effect = [
        {"planned_sections": [_planned(4, "development", "Nouveau")], **same},
        same,
    ]

    result = await generate_more_sections(_plan_row(), current, gemini_client)

    assert result.next_steps is not None
    assert result.next_steps.subtopics == ["Piste B"]   # « Nouveau » vient d'être développée


@pytest.mark.asyncio
async def test_more_sections_next_steps_missing_falls_back_after_retry_failure(gemini_client):
    gemini_client.format_structured.side_effect = [
        {"planned_sections": [_planned(4, "development", "Nouveau")]},   # next_steps absent
        GeminiInvalidResponseError("x"),
    ]

    result = await generate_more_sections(_plan_row(), _plan_with_next_steps(), gemini_client)

    assert len(result.sections) == 1
    assert result.next_steps is not None and result.next_steps.type == "next_steps"


@pytest.mark.asyncio
async def test_more_sections_all_leads_covered_uses_retry(gemini_client):
    gemini_client.format_structured.side_effect = [
        {
            "planned_sections": [_planned(4, "development", "Nouveau")],
            "next_steps": _planned(9, "next_steps", "Suite", ["Piste A", "Nouveau", "Principe"]),
        },
        {"next_steps": _planned(9, "next_steps", "Suite", ["Autre piste"])},
    ]

    result = await generate_more_sections(_plan_row(), _plan_with_next_steps(), gemini_client)

    assert result.next_steps is not None and result.next_steps.subtopics == ["Autre piste"]


@pytest.mark.asyncio
async def test_more_sections_without_new_section_raises(gemini_client):
    current = _api_sections(["Principe"])
    gemini_client.format_structured.return_value = {"planned_sections": [_planned(1, "development", "Principe")]}

    with pytest.raises(GeminiInvalidResponseError):
        await generate_more_sections(_plan_row(), current, gemini_client)
