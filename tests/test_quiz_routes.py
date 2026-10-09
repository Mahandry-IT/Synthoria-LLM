import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import quiz_routes
from app.services import quiz_bank
from app.services.quiz_bank import StartedAttempt, SubmittedAttempt

SID = uuid.uuid4()
AID = uuid.uuid4()

_PUBLIC_Q = {
    "question": "Q ?", "options": ["A", "B"], "correct_option_indices": [], "difficulty": "normale",
    "points": 20.0, "explanation": "", "explanation_per_choice": [], "section_refs": [], "time_limit_seconds": 45,
}


def _settings(**overrides):
    values = {"quiz_attempt_rate_limit_per_minute": 100, "quiz_submit_rate_limit_per_minute": 100, **overrides}
    return SimpleNamespace(**values)


@pytest.fixture
def env(monkeypatch):
    app = FastAPI()
    app.include_router(quiz_routes.router)

    @asynccontextmanager
    async def factory():
        yield object()

    app.state.db_session_factory = factory
    app.state.gemini_client = AsyncMock()
    app.dependency_overrides[quiz_routes.get_settings] = lambda: _settings()
    refill = AsyncMock()
    monkeypatch.setattr(quiz_bank, "refill_bank", refill)
    return SimpleNamespace(app=app, client=TestClient(app), refill=refill)


def test_start_returns_attempt_and_questions_without_answers_then_schedules_refill(env, monkeypatch):
    monkeypatch.setattr(
        quiz_bank, "start_attempt",
        AsyncMock(return_value=StartedAttempt(attempt_id=AID, questions=[_PUBLIC_Q], refill_needed=True)),
    )

    res = env.client.post(f"/courses/{SID}/quiz/attempts")

    assert res.status_code == 201
    body = res.json()
    assert body["attempt_id"] == str(AID)
    assert body["questions"][0]["correct_option_indices"] == [] and body["questions"][0]["points"] == 20.0
    env.refill.assert_awaited_once()
    assert env.refill.await_args.args[0] == SID


def test_start_does_not_schedule_a_second_refill_while_one_is_running(env, monkeypatch):
    monkeypatch.setattr(
        quiz_bank, "start_attempt",
        AsyncMock(return_value=StartedAttempt(attempt_id=AID, questions=[_PUBLIC_Q], refill_needed=True)),
    )
    env.app.state.quiz_refills_in_flight = {SID}

    env.client.post(f"/courses/{SID}/quiz/attempts")

    env.refill.assert_not_called()


@pytest.mark.parametrize("error", [quiz_bank.QuizSessionNotFound, quiz_bank.QuizUnavailable])
def test_start_unknown_session_or_course_without_quiz_is_404(env, monkeypatch, error):
    monkeypatch.setattr(quiz_bank, "start_attempt", AsyncMock(side_effect=error()))
    assert env.client.post(f"/courses/{SID}/quiz/attempts").status_code == 404


def test_start_is_rate_limited(env, monkeypatch):
    env.app.dependency_overrides[quiz_routes.get_settings] = lambda: _settings(quiz_attempt_rate_limit_per_minute=1)
    monkeypatch.setattr(
        quiz_bank, "start_attempt",
        AsyncMock(return_value=StartedAttempt(attempt_id=AID, questions=[], refill_needed=False)),
    )
    env.client.post(f"/courses/{SID}/quiz/attempts")

    res = env.client.post(f"/courses/{SID}/quiz/attempts")

    assert res.status_code == 429


def test_submit_returns_score_status_and_results(env, monkeypatch):
    result = {
        "question": "Q ?", "answer": [0], "correct_option_indices": [0], "is_correct": True,
        "points": 20.0, "points_earned": 20.0, "explanation": "Parce que.", "explanation_per_choice": [],
    }
    submit = AsyncMock(return_value=SubmittedAttempt(score=20.0, max_score=20.0, status="completed", results=[result]))
    monkeypatch.setattr(quiz_bank, "submit_attempt", submit)

    res = env.client.post(f"/courses/{SID}/quiz/attempts/{AID}/submit", json={"answers": [[0]]})

    assert res.status_code == 200
    assert res.json() == {"score": 20.0, "max_score": 20.0, "status": "completed", "results": [result]}
    assert submit.await_args.kwargs == {"aborted": False, "abort_reason": None}


def test_submit_aborted_attempt_returns_null_score(env, monkeypatch):
    submit = AsyncMock(return_value=SubmittedAttempt(score=None, max_score=20.0, status="aborted", results=[]))
    monkeypatch.setattr(quiz_bank, "submit_attempt", submit)

    res = env.client.post(
        f"/courses/{SID}/quiz/attempts/{AID}/submit", json={"answers": [], "aborted": True, "abort_reason": "fullscreen_exit"}
    )

    assert res.json()["score"] is None and res.json()["status"] == "aborted"
    assert submit.await_args.kwargs == {"aborted": True, "abort_reason": "fullscreen_exit"}


@pytest.mark.parametrize(
    "error, code",
    [
        (quiz_bank.QuizAttemptNotFound(), 404),
        (quiz_bank.QuizAttemptAlreadySubmitted(), 409),
        (quiz_bank.InvalidQuizAnswers("2 réponses attendues, 1 reçues"), 422),
    ],
)
def test_submit_error_mapping(env, monkeypatch, error, code):
    monkeypatch.setattr(quiz_bank, "submit_attempt", AsyncMock(side_effect=error))
    assert env.client.post(f"/courses/{SID}/quiz/attempts/{AID}/submit", json={"answers": [[0]]}).status_code == code


@pytest.mark.parametrize("body", [{}, {"answers": [[-1]]}, {"answers": [["x"]]}, {"answers": [], "abort_reason": "x" * 201}])
def test_submit_body_validation_is_422(env, body):
    assert env.client.post(f"/courses/{SID}/quiz/attempts/{AID}/submit", json=body).status_code == 422


def test_history_lists_attempts(env, monkeypatch):
    from datetime import datetime, timezone

    attempt = SimpleNamespace(
        id=AID, started_at=datetime(2026, 10, 9, tzinfo=timezone.utc), finished_at=None,
        score=None, max_score=20.0, status="in_progress", abort_reason=None,
    )
    monkeypatch.setattr(quiz_bank, "list_attempts", AsyncMock(return_value=[attempt]))

    res = env.client.get(f"/courses/{SID}/quiz/attempts")

    assert res.status_code == 200
    assert res.json()["attempts"][0]["attempt_id"] == str(AID)
    assert res.json()["attempts"][0]["status"] == "in_progress"


def test_history_unknown_session_is_404(env, monkeypatch):
    monkeypatch.setattr(quiz_bank, "list_attempts", AsyncMock(side_effect=quiz_bank.QuizSessionNotFound()))
    assert env.client.get(f"/courses/{SID}/quiz/attempts").status_code == 404
