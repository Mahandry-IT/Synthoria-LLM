import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import routes
from app.api.schemas import ApiPlannedSection
from app.core.exceptions import GeminiInvalidResponseError
from app.services.course_plan_generator import MoreSectionsResult


def _section(order: int, type_: str, title: str, subtopics: list[str] | None = None) -> dict:
    return {"type": type_, "title": title, "objective": f"Objectif {title}", "subtopics": subtopics or ["a"], "order": order}


@pytest.fixture
def env(monkeypatch):
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[routes.get_gemini_client] = lambda: AsyncMock()
    monkeypatch.setattr(routes, "_get_active_plan", AsyncMock(return_value=SimpleNamespace(id=uuid.uuid4())))
    return SimpleNamespace(client=TestClient(app), monkeypatch=monkeypatch)


def _payload() -> dict:
    return {
        "plan_id": str(uuid.uuid4()),
        "sections": [_section(1, "development", "Principe"), _section(2, "next_steps", "Suite", ["Piste A"])],
    }


def test_more_sections_returns_refreshed_next_steps(env):
    result = MoreSectionsResult(
        sections=[ApiPlannedSection(**_section(3, "development", "Nouveau"))],
        next_steps=ApiPlannedSection(**_section(2, "next_steps", "Suite", ["Piste neuve"])),
    )
    generate = AsyncMock(return_value=result)
    env.monkeypatch.setattr(routes, "generate_more_sections", generate)

    res = env.client.post("/courses/plan/more-sections", json=_payload())

    assert res.status_code == 200
    body = res.json()
    assert [s["title"] for s in body["sections"]] == ["Nouveau"]
    assert (body["next_steps"]["title"], body["next_steps"]["order"]) == ("Suite", 2)
    assert body["next_steps"]["subtopics"] == ["Piste neuve"]
    assert len(generate.await_args.kwargs["current_sections"]) == 2


def test_more_sections_next_steps_is_null_when_not_refreshed(env):
    result = MoreSectionsResult(sections=[ApiPlannedSection(**_section(3, "development", "Nouveau"))])
    env.monkeypatch.setattr(routes, "generate_more_sections", AsyncMock(return_value=result))

    body = env.client.post("/courses/plan/more-sections", json=_payload()).json()

    assert body["next_steps"] is None and len(body["sections"]) == 1


def test_more_sections_invalid_gemini_response_is_502(env):
    env.monkeypatch.setattr(
        routes, "generate_more_sections", AsyncMock(side_effect=GeminiInvalidResponseError("x"))
    )
    assert env.client.post("/courses/plan/more-sections", json=_payload()).status_code == 502
