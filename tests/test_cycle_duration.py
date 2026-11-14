"""Durée estimée d'une section au format cycle : calculée côté serveur (aucun appel Gemini), exposée dans
`CourseSection.cycle` (`minutes_min` / `minutes_max`), absente pour les anciennes sections."""

import random
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import routes
from app.api.schemas import ApiSectionCycle, CourseGenerationResponse, CourseSection
from app.schemas.course_generation import Section
from app.services.course_depth import estimate_minutes, get_profile, render_cycle_rules
from app.services.course_generator import INCOMPLETE_SECTION_NOTICE, _map_sections_to_course_sections, with_depth
from app.services.cycle_mapping import with_cycle_minutes
from tests.cycle_fixtures import cycle_section, old_format_section


def _steps(count: int):
    def mutate(section: dict) -> None:
        steps = section["cycle"]["steps"]
        while len(steps) < count:
            steps.append({**steps[-1], "support": {**steps[-1]["support"], "kind": "order"}})
        section["cycle"]["steps"] = steps[:count]

    return mutate


def _mapped(depth: str | None = None, count: int = 4, title: str = "Sous-réseaux") -> list[CourseSection]:
    return _map_sections_to_course_sections(
        [Section.model_validate(cycle_section(title, _steps(count)))], rng=random.Random(0), depth=depth
    )


# ─── Calcul ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("depth", "steps", "expected"),
    [
        ("express", 3, (3, 3)),
        ("standard", 4, (5, 5)),  # base du mode standard
        ("standard", 3, (4, 4)),  # une mini-étape de moins : une minute de moins
        ("standard", 5, (6, 6)),
        ("approfondi", 5, (8, 10)),
        ("approfondi", 7, (10, 12)),
        ("approfondi", 6, (9, 11)),
    ],
)
def test_estimate_is_the_mode_base_adjusted_by_the_number_of_steps(depth, steps, expected):
    assert estimate_minutes(depth, steps) == expected


def test_estimate_is_always_a_valid_ordered_range_within_bounds():
    for depth in ("express", "standard", "approfondi"):
        for steps in (0, 1, 3, 8, 100):
            low, high = estimate_minutes(depth, steps)
            assert 1 <= low <= high <= 30


def test_unknown_mode_reads_as_the_legacy_mode():
    assert estimate_minutes("inconnu", 5) == estimate_minutes("approfondi", 5)


def test_the_estimate_is_never_asked_from_the_model():
    assert "calculée côté serveur" in render_cycle_rules(get_profile("standard"))
    assert "minutes_min" not in str(Section.model_json_schema())  # le schéma Gemini ne porte aucune durée


# ─── Exposition ──────────────────────────────────────────────────────────────────────────


def test_mapping_without_a_mode_leaves_the_duration_unset():
    (section,) = _mapped(depth=None)
    assert section.cycle.minutes_min is None and section.cycle.minutes_max is None


def test_mapping_with_a_mode_stamps_the_duration():
    (section,) = _mapped(depth="standard", count=5)
    assert (section.cycle.minutes_min, section.cycle.minutes_max) == (6, 6)


def test_with_depth_stamps_cycle_sections_only_and_never_calls_gemini():
    intro = Section.model_validate({"type": "introduction", "title": "Introduction", "blocks": [{"type": "text", "text": "i"}]})
    old = Section.model_validate(old_format_section("Ancien"))
    cycle = Section.model_validate(cycle_section("Cycle"))
    response = CourseGenerationResponse(
        mode="question_only", format="full_course", sources=[], introduction={},
        meta={"title": "t", "subject": "s", "generated_at": "2026-10-12T10:00:00Z"},
        sections=_map_sections_to_course_sections([intro, old, cycle], rng=random.Random(0)), summary="r",
    )

    stamped = with_depth(response, "standard")  # fonction pure : aucun client Gemini en jeu

    assert stamped.meta.depth == "standard"
    assert [s.cycle is None for s in stamped.sections] == [True, True, False]
    assert (stamped.sections[2].cycle.minutes_min, stamped.sections[2].cycle.minutes_max) == (5, 5)
    assert response.sections[2].cycle.minutes_min is None  # l'original n'est pas muté
    assert stamped.model_dump(mode="json")["sections"][1]["cycle"] is None  # ancienne section : pas de durée


def test_with_depth_on_a_course_without_cycle_is_unchanged():
    old = Section.model_validate(old_format_section("Ancien"))
    response = CourseGenerationResponse(
        mode="question_only", format="full_course", sources=[], introduction={},
        meta={"title": "t", "subject": "s", "generated_at": "2026-10-12T10:00:00Z"},
        sections=_map_sections_to_course_sections([old]), summary="r",
    )
    stamped = with_depth(response, "approfondi")
    assert stamped.sections == response.sections and stamped.sections[0].cycle is None


def test_with_cycle_minutes_leaves_a_section_without_cycle_as_it_is():
    (legacy,) = _map_sections_to_course_sections([Section.model_validate(old_format_section("Ancien"))])
    assert with_cycle_minutes(legacy, "standard") is legacy


def test_stored_cycle_without_duration_stays_valid_and_the_duration_round_trips():
    (section,) = _mapped(depth=None)
    payload = section.cycle.model_dump(mode="json", exclude_none=True)
    assert "minutes_min" not in payload and ApiSectionCycle.model_validate(payload).minutes_min is None

    (stamped,) = _mapped(depth="approfondi", count=5)
    again = ApiSectionCycle.model_validate(stamped.cycle.model_dump(mode="json"))
    assert (again.minutes_min, again.minutes_max) == (8, 10)


# ─── Régénération et ajout : routes ───────────────────────────────────────────────────────


@pytest.fixture
def env(monkeypatch):
    app = FastAPI()
    app.include_router(routes.router)

    @asynccontextmanager
    async def factory():
        yield object()

    app.state.db_session_factory = factory
    app.state.vector_store = AsyncMock()
    app.dependency_overrides[routes.get_gemini_client] = lambda: AsyncMock()
    broken = {
        "id": "0", "title": "Notion en échec", "incomplete": True, "quoi": "", "pourquoi": "", "comment": INCOMPLETE_SECTION_NOTICE,
        "note": "", "worked_example": {"statement": "", "steps": [], "result": ""},
    }
    row = SimpleNamespace(
        id=uuid.uuid4(), question="Q ?", mode="file_question", filenames=[],
        gemini_response={"meta": {"title": "t", "subject": "s", "depth": "standard"}, "sections": [broken], "next_steps": []},
    )
    monkeypatch.setattr(routes.course_session_repository, "get_by_id", AsyncMock(return_value=row))
    return SimpleNamespace(client=TestClient(app), monkeypatch=monkeypatch, row=row)


def test_regenerated_cycle_section_carries_its_duration_and_is_persisted_with_it(env):
    env.monkeypatch.setattr(routes, "regenerate_section", AsyncMock(return_value=Section.model_validate(cycle_section("Notion en échec"))))
    update_section = AsyncMock(return_value=env.row)
    env.monkeypatch.setattr(routes.course_session_repository, "update_section", update_section)

    res = env.client.post(f"/courses/{env.row.id}/sections/0/regenerate")

    assert res.status_code == 200
    cycle = res.json()["cycle"]
    assert (cycle["minutes_min"], cycle["minutes_max"]) == (5, 5)
    assert update_section.await_args.args[3]["cycle"]["minutes_min"] == 5


def test_added_cycle_sections_carry_their_duration(env):
    env.monkeypatch.setattr(
        routes, "add_course_sections",
        AsyncMock(return_value=([Section.model_validate(cycle_section("A")), Section.model_validate(cycle_section("B", _steps(5)))], [])),
    )
    append = AsyncMock(return_value=env.row)
    env.monkeypatch.setattr(routes.course_session_repository, "append_sections", append)

    res = env.client.post(f"/courses/{env.row.id}/sections", json={})

    assert res.status_code == 200
    durations = [(s["cycle"]["minutes_min"], s["cycle"]["minutes_max"]) for s in res.json()["sections"]]
    assert durations == [(5, 5), (6, 6)]
    assert [s["cycle"]["minutes_min"] for s in append.await_args.args[2]] == [5, 6]


def test_added_old_format_section_has_no_cycle_and_no_duration(env):
    env.monkeypatch.setattr(
        routes, "add_course_sections", AsyncMock(return_value=([Section.model_validate(old_format_section("Ancienne"))], [])),
    )
    env.monkeypatch.setattr(routes.course_session_repository, "append_sections", AsyncMock(return_value=env.row))

    res = env.client.post(f"/courses/{env.row.id}/sections", json={})

    assert res.status_code == 200 and res.json()["sections"][0]["cycle"] is None
