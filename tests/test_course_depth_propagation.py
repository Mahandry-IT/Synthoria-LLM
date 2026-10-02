"""Propagation du mode de cours (`depth`) : API → plan persisté (migration 015) → génération par
lots → `meta.depth` de la session → régénération / ajout de sections. Absent = approfondi."""

import importlib.util
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import routes
from app.api.schemas import ApiPlannedSection, CourseMeta
from app.core.config import Settings
from app.db.models import CoursePlan
from app.db.schema_sync import ADDED_COLUMNS, sync_added_columns
from app.repositories import course_plan_repository
from app.schemas.course_generation import CourseGenerationSchema, CoursePlanSchema, SectionsBatchSchema
from app.services.course_plan_generator import generate_course_from_validated_plan, generate_course_plan
from app.services.course_section_adder import add_course_sections
from app.services.section_regenerator import regenerate_section

PLAN_META = {"title": "Transformateurs", "subject": "Électrotechnique", "language": "fr"}


def _conform_section(title: str) -> dict:
    """Section DEVELOPMENT conforme à tous les modes : 4 blocs dont 2 non textuels, peu de mots."""
    return {
        "type": "development", "title": title, "blocks": [],
        "subsections": [
            {"title": "Pourquoi", "blocks": [{"type": "list", "list_items": ["enjeu"]}]},
            {"title": "Quoi", "blocks": [
                {"type": "text", "text": "définition"},
                {"type": "table", "table": {"caption": "c", "headers": ["h"], "rows": [["x"]]}},
            ]},
            {"title": "Comment", "blocks": [{"type": "text", "text": "mécanisme"}]},
        ],
    }


def _text_only_section(title: str) -> dict:
    return {
        "type": "development", "title": title, "blocks": [],
        "subsections": [
            {"title": name, "blocks": [{"type": "text", "text": name.lower()}]}
            for name in ("Pourquoi", "Quoi", "Comment")
        ],
    }


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


def _fake_gemini() -> AsyncMock:
    async def format_structured(raw_answer, system_instruction, *, response_schema=None):
        if response_schema is SectionsBatchSchema:
            count = int(re.search(r"Génère exactement (\d+)", raw_answer).group(1))
            return {"sections": [_conform_section(f"s{i}") for i in range(count)]}
        assert response_schema is CourseGenerationSchema
        return _wrap_up()

    client = AsyncMock()
    client.format_structured.side_effect = format_structured
    return client


def _plan_row(**extra) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(), question="Q", mode="file_question", filenames=[],
        retrieval_context={"chunks": [], "file_sources": [], "search_query": "q", "web_research": None},
        plan={"meta": PLAN_META, "planned_sections": [], "coverage_notes": ""},
        **extra,
    )


def _sections() -> list[ApiPlannedSection]:
    return [
        ApiPlannedSection(type="introduction", title="Introduction", order=1),
        ApiPlannedSection(type="development", title="Principe", order=2),
    ]


def _prompts(client: AsyncMock, schema) -> list[str]:
    return [c.kwargs["raw_answer"] for c in client.format_structured.call_args_list if c.kwargs["response_schema"] is schema]


# ─── Persistance : modèle, repository, migration ─────────────────────────────


@pytest.mark.asyncio
async def test_repository_persists_depth_and_defaults_to_approfondi():
    session = AsyncMock()
    session.add = MagicMock()
    common = dict(
        question="Q", mode="question_only", filenames=[], top_k=6, full_document=False,
        retrieval_context={}, plan={}, expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )

    express = await course_plan_repository.save(session, depth="express", **common)
    legacy = await course_plan_repository.save(session, **common)

    assert isinstance(express, CoursePlan) and express.depth == "express"
    assert legacy.depth == "approfondi"


def test_course_plan_model_has_server_default_for_existing_rows():
    column = CoursePlan.__table__.c.depth
    assert column.nullable is False and column.server_default.arg == "approfondi"


def test_migration_015_adds_depth_idempotently(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "migrations" / "versions" / "015_add_course_plan_depth.py"
    spec = importlib.util.spec_from_file_location("migration_015", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    executed: list[str] = []
    monkeypatch.setattr(module.op, "execute", executed.append, raising=False)

    module.upgrade()
    module.downgrade()

    assert (module.revision, module.down_revision) == ("015", "014")
    assert "ADD COLUMN IF NOT EXISTS depth VARCHAR(12) NOT NULL DEFAULT 'approfondi'" in executed[0]
    assert "DROP COLUMN IF EXISTS depth" in executed[1]
    assert executed[0] in ADDED_COLUMNS  # même instruction que la synchronisation au démarrage


@pytest.mark.asyncio
async def test_startup_schema_sync_executes_every_added_column():
    conn = AsyncMock()
    await sync_added_columns(conn)
    assert conn.execute.await_count == len(ADDED_COLUMNS)


# ─── Services ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_generate_plan_injects_section_count_of_the_mode():
    gemini = AsyncMock()
    gemini.reformulate_query.return_value = "q"
    gemini.format_structured.return_value = {
        "meta": PLAN_META,
        "planned_sections": [{"type": "development", "title": "A", "objective": "o", "order": 1}],
    }
    store = AsyncMock()
    store.search.return_value = []
    settings = Settings(gemini_api_key="k", gemini_use_search_grounding=False)

    plan, _ = await generate_course_plan("Q", store, gemini, settings, mode="file_question", depth="express")

    assert isinstance(plan, CoursePlanSchema)
    assert "entre 5 et 8 sections" in gemini.format_structured.await_args.kwargs["raw_answer"]


@pytest.mark.asyncio
async def test_from_plan_applies_mode_rules_and_writes_meta_depth():
    client = _fake_gemini()
    settings = Settings(gemini_api_key="k", course_plan_batch_size=4)

    result = await generate_course_from_validated_plan(_plan_row(depth="express"), _sections(), client, settings)

    assert result.meta.depth == "express"
    assert "150 mots" in _prompts(client, SectionsBatchSchema)[0]
    assert "5 à 6 questions" in _prompts(client, CourseGenerationSchema)[0]


@pytest.mark.asyncio
async def test_from_plan_without_depth_is_approfondi():
    client = _fake_gemini()
    settings = Settings(gemini_api_key="k", course_plan_batch_size=4)

    result = await generate_course_from_validated_plan(_plan_row(), _sections(), client, settings)

    assert result.meta.depth == "approfondi"
    assert "500 mots" in _prompts(client, SectionsBatchSchema)[0]


def test_course_meta_of_legacy_session_reads_as_approfondi():
    legacy = {"title": "t", "subject": "s", "generated_at": "2026-01-01"}
    assert CourseMeta.model_validate(legacy).depth == "approfondi"


def _session_row(depth: str | None) -> SimpleNamespace:
    meta = {"title": "t", "subject": "s"} | ({"depth": depth} if depth else {})
    return SimpleNamespace(
        question="Q ?", mode="file_question", filenames=[],
        gemini_response={"meta": meta, "sections": [], "next_steps": []},
    )


def _regen_gemini(*responses: dict) -> AsyncMock:
    gemini = AsyncMock()
    gemini.reformulate_query.return_value = "q"
    gemini.format_structured.side_effect = [{"sections": [r]} for r in responses]
    return gemini


def _store() -> AsyncMock:
    store = AsyncMock()
    store.search.return_value = []
    return store


@pytest.mark.asyncio
async def test_regenerate_section_reads_meta_depth_and_returns_conform_section_without_repair():
    gemini = _regen_gemini(_conform_section("x"))

    result = await regenerate_section(_session_row("express"), "Notion", gemini, _store(), Settings(gemini_api_key="k"))

    assert result.title == "Notion"
    assert gemini.format_structured.await_count == 1
    assert "150 mots" in gemini.format_structured.await_args.kwargs["raw_answer"]


@pytest.mark.asyncio
async def test_regenerate_section_repairs_non_conform_section_and_keeps_the_best():
    gemini = _regen_gemini(_text_only_section("x"), _conform_section("y"))

    result = await regenerate_section(_session_row("standard"), "Notion", gemini, _store(), Settings(gemini_api_key="k"))

    assert gemini.format_structured.await_count == 2
    repair_prompt = gemini.format_structured.await_args.kwargs["raw_answer"]
    assert "aucun bloc visuel" in repair_prompt and "300 mots" in repair_prompt
    assert result.title == "Notion"
    assert any(b.type.value == "table" for sub in result.subsections for b in sub.blocks)


@pytest.mark.asyncio
async def test_regenerate_section_repair_is_bounded_and_keeps_original_when_never_better():
    still_bad = _text_only_section("x")
    gemini = _regen_gemini(still_bad, still_bad, still_bad, still_bad)

    result = await regenerate_section(_session_row(None), "Notion", gemini, _store(), Settings(gemini_api_key="k"))

    assert gemini.format_structured.await_count == 3  # 1 génération + _MAX_REPAIR_ATTEMPTS (2)
    assert "500 mots" in gemini.format_structured.call_args_list[0].kwargs["raw_answer"]  # absent = approfondi
    assert result.title == "Notion"


@pytest.mark.asyncio
async def test_add_sections_injects_rules_of_the_course_mode():
    gemini = _regen_gemini(_conform_section("Nouvelle"))

    await add_course_sections(_session_row("express"), "Sujet", gemini, _store(), Settings(gemini_api_key="k"))

    assert "Règles du mode « express »" in gemini.format_structured.await_args.kwargs["raw_answer"]


# ─── Routes ──────────────────────────────────────────────────────────────────


@pytest.fixture
def api(monkeypatch):
    app = FastAPI()
    app.include_router(routes.router)

    @asynccontextmanager
    async def factory():
        yield object()

    app.state.db_session_factory = factory
    app.state.vector_store = AsyncMock()
    app.dependency_overrides[routes.get_gemini_client] = lambda: MagicMock()
    plan = CoursePlanSchema.model_validate({
        "meta": PLAN_META,
        "planned_sections": [{"type": "development", "title": "A", "objective": "o", "order": 1}],
    })
    generate_plan = AsyncMock(return_value=(plan, {"chunks": []}))
    monkeypatch.setattr(routes, "generate_course_plan", generate_plan)

    async def save(db, **kwargs):
        return SimpleNamespace(id=uuid.uuid4(), expires_at=kwargs["expires_at"], depth=kwargs.get("depth"))

    save_mock = AsyncMock(side_effect=save)
    monkeypatch.setattr(routes.course_plan_repository, "save", save_mock)
    return SimpleNamespace(client=TestClient(app), generate_plan=generate_plan, save=save_mock, plan=plan,
                           monkeypatch=monkeypatch)


def test_create_plan_persists_and_returns_requested_depth(api):
    res = api.client.post("/courses/plan", json={"question": "Q", "depth": "express"})

    assert res.status_code == 200
    assert res.json()["depth"] == "express"
    assert api.save.await_args.kwargs["depth"] == "express"
    assert api.generate_plan.await_args.kwargs["depth"] == "express"


def test_create_plan_without_depth_defaults_to_approfondi(api):
    res = api.client.post("/courses/plan", json={"question": "Q"})

    assert res.json()["depth"] == "approfondi"
    assert api.save.await_args.kwargs["depth"] == "approfondi"


def test_create_plan_rejects_unknown_depth(api):
    assert api.client.post("/courses/plan", json={"question": "Q", "depth": "turbo"}).status_code == 422


def test_get_plan_detail_returns_persisted_depth_and_legacy_default(api):
    row = SimpleNamespace(
        id=uuid.uuid4(), expires_at=datetime.now(timezone.utc) + timedelta(hours=1), mode="question_only",
        plan=api.plan.model_dump(mode="json"), question="Q", filenames=[], depth="standard",
    )
    api.monkeypatch.setattr(routes.course_plan_repository, "get_by_id", AsyncMock(return_value=row))
    assert api.client.get(f"/courses/plans/{row.id}").json()["depth"] == "standard"

    del row.depth  # plan lu avant la migration / objet sans colonne
    assert api.client.get(f"/courses/plans/{row.id}").json()["depth"] == "approfondi"


def test_generate_course_forwards_depth(api):
    from app.api.schemas import CourseGenerationResponse

    response = CourseGenerationResponse(
        mode="question_only", format="focused_answer",
        meta={"title": "t", "subject": "s", "generated_at": "2026-10-02", "depth": "standard"},
        sources=[], answer={"summary": "s"}, summary="",
    )
    generate = AsyncMock(return_value=response)
    api.monkeypatch.setattr(routes, "generate_course_from_question", generate)
    api.monkeypatch.setattr(routes, "_persist_course_session", AsyncMock(return_value=None))

    res = api.client.post("/courses/generate", json={"question": "Q", "depth": "standard"})

    assert res.status_code == 200
    assert generate.await_args.kwargs["depth"] == "standard"
    assert res.json()["meta"]["depth"] == "standard"
