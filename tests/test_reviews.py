import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import review_routes
from app.schemas.course_generation import Section
from app.services.course_generator import _map_sections_to_course_sections
from app.services.leitner import flashcards_from_course, next_review

INTERVALS = [1, 3, 7, 21]
NOW = datetime(2026, 9, 22, tzinfo=timezone.utc)


def test_leitner_success_climbs_boxes_and_caps():
    box, due = next_review(None, True, NOW, INTERVALS)
    assert (box, due) == (0, NOW + timedelta(days=1))
    box, due = next_review(box, True, NOW, INTERVALS)
    assert (box, due) == (1, NOW + timedelta(days=3))
    assert next_review(2, True, NOW, INTERVALS) == (3, NOW + timedelta(days=21))
    assert next_review(3, True, NOW, INTERVALS) == (3, NOW + timedelta(days=21))  # plafonné


def test_leitner_failure_resets_to_first_box():
    assert next_review(3, False, NOW, INTERVALS) == (0, NOW + timedelta(days=1))
    assert next_review(None, False, NOW, INTERVALS) == (0, NOW + timedelta(days=1))


def _course() -> dict:
    section = Section.model_validate({
        "type": "development", "title": "T", "blocks": [],
        "subsections": [{"title": "Quoi", "blocks": [{"type": "text", "text": "x"}]}],
        "check_questions": [{
            "question": "Q ?", "choices": ["A", "B"], "correct_indices": [1], "difficulty": "facile",
            "explanation": "Parce que.", "requires_calculation": False,
        }],
    })
    api = _map_sections_to_course_sections([section])
    return {"meta": {"title": "Cours"}, "sections": [s.model_dump(mode="json") for s in api]}


def test_flashcards_derived_from_check_questions():
    cards = flashcards_from_course(_course())

    assert cards == [{"card_id": "0-0", "front": "Q ?", "back": "B\nParce que.", "section_ref": "0"}]


@pytest.fixture
def env(monkeypatch):
    app = FastAPI()
    app.include_router(review_routes.router)

    @asynccontextmanager
    async def factory():
        yield object()

    app.state.db_session_factory = factory
    sid = uuid.uuid4()
    row = SimpleNamespace(id=sid, gemini_response=_course())
    repo = review_routes.course_session_repository
    monkeypatch.setattr(repo, "list_paginated", AsyncMock(return_value=([row], 1)))
    monkeypatch.setattr(repo, "get_by_id", AsyncMock(side_effect=lambda db, i: row if i == sid else None))
    reviews = {}
    rr = review_routes.flashcard_review_repository
    monkeypatch.setattr(rr, "get_for_sessions", AsyncMock(side_effect=lambda db, ids: dict(reviews)))
    monkeypatch.setattr(rr, "get_one", AsyncMock(side_effect=lambda db, s, c: reviews.get((s, c))))

    async def upsert(db, *, session_id, card_id, box, due_at, last_result):
        reviews[(session_id, card_id)] = SimpleNamespace(box=box, due_at=due_at, last_result=last_result)

    monkeypatch.setattr(rr, "upsert", upsert)
    return SimpleNamespace(client=TestClient(app), sid=sid, reviews=reviews)


def test_new_cards_are_due_then_scheduled_after_review(env):
    due = env.client.get("/reviews/due").json()
    assert due["total_due"] == 1 and due["cards"][0]["card_id"] == "0-0" and due["cards"][0]["due_at"] is None

    res = env.client.post(f"/reviews/{env.sid}/0-0", json={"result": "correct"})
    assert res.status_code == 200 and res.json()["box"] == 0

    assert env.client.get("/reviews/due").json()["total_due"] == 0  # échéance J+1


def test_overdue_card_comes_back_and_failure_resets(env):
    env.reviews[(env.sid, "0-0")] = SimpleNamespace(box=2, due_at=NOW - timedelta(days=365), last_result="correct")

    due = env.client.get("/reviews/due").json()
    assert due["cards"][0]["box"] == 2

    res = env.client.post(f"/reviews/{env.sid}/0-0", json={"result": "incorrect"})
    assert res.json()["box"] == 0


def test_review_unknown_card_or_session_is_404_and_bad_result_422(env):
    assert env.client.post(f"/reviews/{env.sid}/9-9", json={"result": "correct"}).status_code == 404
    assert env.client.post(f"/reviews/{uuid.uuid4()}/0-0", json={"result": "correct"}).status_code == 404
    assert env.client.post(f"/reviews/{env.sid}/0-0", json={"result": "peut-être"}).status_code == 422
