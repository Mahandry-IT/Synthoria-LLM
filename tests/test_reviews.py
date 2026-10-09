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
from app.services.leitner import flashcards_from_course, next_review, resolve_variant_no, review_mode

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

    assert cards == [{
        "card_id": "0-0", "front": "Q ?", "back": "B\nParce que.", "section_ref": "0",
        "choices": ["A", "B"], "correct_indices": [1], "explanation": "Parce que.",
    }]


@pytest.fixture
def env(monkeypatch):
    app = FastAPI()
    app.include_router(review_routes.router)

    @asynccontextmanager
    async def factory():
        yield object()

    app.state.db_session_factory = factory
    app.state.gemini_client = AsyncMock()
    sid = uuid.uuid4()
    row = SimpleNamespace(id=sid, gemini_response=_course())
    repo = review_routes.course_session_repository
    monkeypatch.setattr(repo, "list_paginated", AsyncMock(return_value=([row], 1)))
    monkeypatch.setattr(repo, "get_by_id", AsyncMock(side_effect=lambda db, i: row if i == sid else None))
    reviews = {}
    rr = review_routes.flashcard_review_repository
    monkeypatch.setattr(rr, "get_for_sessions", AsyncMock(side_effect=lambda db, ids: dict(reviews)))
    monkeypatch.setattr(rr, "get_one", AsyncMock(side_effect=lambda db, s, c: reviews.get((s, c))))

    async def upsert(db, *, session_id, card_id, box, due_at, last_result, variant_no=0):
        reviews[(session_id, card_id)] = SimpleNamespace(
            box=box, due_at=due_at, last_result=last_result, variant_no=variant_no
        )

    monkeypatch.setattr(rr, "upsert", upsert)
    variants = {}
    monkeypatch.setattr(
        review_routes.flashcard_variant_repository, "get_for_sessions", AsyncMock(side_effect=lambda db, ids: variants)
    )
    generate = AsyncMock()
    monkeypatch.setattr(review_routes.flashcard_variants, "generate_variants", generate)
    return SimpleNamespace(client=TestClient(app), app=app, sid=sid, reviews=reviews, variants=variants, generate=generate)


def test_new_cards_are_due_then_scheduled_after_review(env):
    due = env.client.get("/reviews/due").json()
    assert due["total_due"] == 1 and due["cards"][0]["card_id"] == "0-0" and due["cards"][0]["due_at"] is None

    res = env.client.post(f"/reviews/{env.sid}/0-0", json={"result": "correct"})
    assert res.status_code == 200 and res.json()["box"] == 0

    assert env.client.get("/reviews/due").json()["total_due"] == 0  # échéance J+1


def test_overdue_card_comes_back_and_failure_resets(env):
    env.reviews[(env.sid, "0-0")] = SimpleNamespace(box=2, due_at=NOW - timedelta(days=365), last_result="correct", variant_no=0)

    due = env.client.get("/reviews/due").json()
    assert due["cards"][0]["box"] == 2

    res = env.client.post(f"/reviews/{env.sid}/0-0", json={"result": "incorrect"})
    assert res.json()["box"] == 0


def test_review_unknown_card_or_session_is_404_and_bad_result_422(env):
    assert env.client.post(f"/reviews/{env.sid}/9-9", json={"result": "correct"}).status_code == 404
    assert env.client.post(f"/reviews/{uuid.uuid4()}/0-0", json={"result": "correct"}).status_code == 404
    assert env.client.post(f"/reviews/{env.sid}/0-0", json={"result": "peut-être"}).status_code == 422


# ─── Variantes et mode mixte ─────────────────────────────────────────────────


def _variant(variant_no: int, front: str = "Variante ?") -> SimpleNamespace:
    return SimpleNamespace(
        variant_no=variant_no, front=front, choices=["X", "Y", "Z"], correct_indices=[2], explanation="Car Z."
    )


def _due_review(box: int, variant_no: int) -> SimpleNamespace:
    return SimpleNamespace(box=box, due_at=NOW - timedelta(days=1), last_result="correct", variant_no=variant_no)


def test_new_card_is_variant_zero_in_qcm_mode_with_choices(env):
    card = env.client.get("/reviews/due").json()["cards"][0]

    assert card["variant_no"] == 0 and card["mode"] == "qcm"
    assert card["choices"] == ["A", "B"] and card["correct_indices"] == [1] and card["explanation"] == "Parce que."
    assert card["back"] == "B\nParce que."


def test_due_card_shows_its_generated_variant_in_alternating_mode(env):
    env.reviews[(env.sid, "0-0")] = _due_review(box=0, variant_no=1)
    env.variants[(env.sid, "0-0")] = {1: _variant(1)}

    card = env.client.get("/reviews/due").json()["cards"][0]

    assert card["variant_no"] == 1 and card["front"] == "Variante ?"
    assert card["mode"] == "text"  # box 0 + variante 1 : impair
    assert card["choices"] == ["X", "Y", "Z"] and card["correct_indices"] == [2] and card["back"] == "Z\nCar Z."


def test_missing_next_variant_falls_back_to_the_current_one(env):
    env.reviews[(env.sid, "0-0")] = _due_review(box=1, variant_no=1)

    card = env.client.get("/reviews/due").json()["cards"][0]

    assert card["variant_no"] == 0 and card["front"] == "Q ?" and card["mode"] == "text"  # box 1 + 0


def test_correct_advances_variant_and_schedules_generation_when_missing(env):
    res = env.client.post(f"/reviews/{env.sid}/0-0", json={"result": "correct"})

    assert res.json()["variant_no"] == 1
    assert env.reviews[(env.sid, "0-0")].variant_no == 1
    env.generate.assert_awaited_once()
    assert env.generate.await_args.args[0] == env.sid


def test_correct_with_existing_next_variant_does_not_call_gemini(env):
    env.variants[(env.sid, "0-0")] = {1: _variant(1)}

    res = env.client.post(f"/reviews/{env.sid}/0-0", json={"result": "correct"})

    assert res.json()["variant_no"] == 1
    env.generate.assert_not_called()


def test_generation_already_running_for_the_course_is_not_scheduled_twice(env):
    env.app.state.flashcard_variants_in_flight = {env.sid}

    env.client.post(f"/reviews/{env.sid}/0-0", json={"result": "correct"})

    env.generate.assert_not_called()


def test_incorrect_keeps_the_current_variant(env):
    env.reviews[(env.sid, "0-0")] = _due_review(box=2, variant_no=1)
    env.variants[(env.sid, "0-0")] = {1: _variant(1)}

    res = env.client.post(f"/reviews/{env.sid}/0-0", json={"result": "incorrect"})

    assert res.json()["variant_no"] == 1 and res.json()["box"] == 0
    env.generate.assert_not_called()


@pytest.mark.parametrize(
    "box, variant_no, has_choices, expected",
    [(0, 0, True, "qcm"), (1, 0, True, "text"), (1, 1, True, "qcm"), (0, 0, False, "text")],
)
def test_review_mode_alternates_on_box_plus_variant(box, variant_no, has_choices, expected):
    assert review_mode(box, variant_no, has_choices) == expected


@pytest.mark.parametrize("stored, available, expected", [(0, set(), 0), (2, {1}, 1), (2, {1, 2, 3}, 2), (1, set(), 0)])
def test_resolve_variant_no_falls_back_to_highest_existing(stored, available, expected):
    assert resolve_variant_no(stored, available) == expected
