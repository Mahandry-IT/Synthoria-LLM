import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import exercise_routes as routes
from app.api import routes as core_routes
from app.core.errors import register_error_handlers
from app.core.exceptions import GeminiInvalidResponseError, GeminiQuotaExceededError
from tests.exercise_fixtures import course, exercise_row, job_row

SID = uuid.uuid4()


@pytest.fixture
def env(monkeypatch):
    app = FastAPI()
    register_error_handlers(app, include_debug=False)
    app.include_router(routes.router)

    @asynccontextmanager
    async def factory():
        yield object()

    app.state.db_session_factory = factory
    gemini = AsyncMock()
    gemini.format_structured.return_value = {
        "feedback": "Bien joué.", "correct_points": ["Série"], "missing_points": ["Unité"],
        "expected_approach": "Additionner les résistances.",
    }
    app.dependency_overrides[core_routes.get_gemini_client] = lambda: gemini

    ns = SimpleNamespace(
        client=TestClient(app), app=app, gemini=gemini, monkeypatch=monkeypatch,
        course=AsyncMock(return_value=SimpleNamespace(gemini_response=course())),
        exists=AsyncMock(return_value=True),
        job=AsyncMock(return_value=None),
        exercises=AsyncMock(return_value=[]),
        one=AsyncMock(return_value=exercise_row()),
        enqueue=AsyncMock(return_value=job_row("pending")),
    )
    monkeypatch.setattr(routes.course_session_repository, "get_by_id", ns.course)
    monkeypatch.setattr(routes.quiz_repository, "session_exists", ns.exists)
    monkeypatch.setattr(routes.exercise_job_repository, "get_by_session", ns.job)
    monkeypatch.setattr(routes.exercise_repository, "list_by_session", ns.exercises)
    monkeypatch.setattr(routes.exercise_repository, "get_for_session", ns.one)
    monkeypatch.setattr(routes, "enqueue_exercise_job", ns.enqueue)
    return ns


def _base(session_id=SID) -> str:
    return f"/courses/{session_id}/exercises"


# --- génération ---------------------------------------------------------------------------------

def test_generate_returns_202_with_status(env):
    res = env.client.post(f"{_base()}/generate")
    assert res.status_code == 202 and res.json() == {"status": "pending"}
    env.enqueue.assert_awaited_once()


@pytest.mark.parametrize("status", ["generating", "done"])
def test_generate_is_idempotent_and_reports_existing_state(env, status):
    env.enqueue.return_value = job_row(status)
    res = env.client.post(f"{_base()}/generate")
    assert res.status_code == 202 and res.json() == {"status": status}


def test_generate_unknown_session_is_404(env):
    env.course.return_value = None
    assert env.client.post(f"{_base()}/generate").status_code == 404
    env.enqueue.assert_not_awaited()


def test_generate_incomplete_course_is_409(env):
    env.course.return_value = SimpleNamespace(gemini_response=course(sections=[{"id": "s1", "title": "Vide"}]))
    res = env.client.post(f"{_base()}/generate")
    assert res.status_code == 409
    env.enqueue.assert_not_awaited()


def test_generate_without_quiz_is_409(env):
    env.course.return_value = SimpleNamespace(gemini_response=course(quiz=False))
    assert env.client.post(f"{_base()}/generate").status_code == 409
    env.enqueue.assert_not_awaited()


def test_generate_is_rate_limited(env):
    env.app.dependency_overrides[routes.get_settings] = lambda: SimpleNamespace(
        exercise_generate_rate_limit_per_minute=2, exercise_answer_rate_limit_per_minute=10,
        exercise_answer_max_length=4000,
    )
    codes = [env.client.post(f"{_base()}/generate").status_code for _ in range(3)]
    assert codes == [202, 202, 429]


# --- liste et détail ----------------------------------------------------------------------------

def test_list_absent_when_no_job(env):
    res = env.client.get(_base())
    assert res.status_code == 200
    assert res.json() == {"status": "absent", "error_message": None, "exercises": []}


def test_list_unknown_session_is_404(env):
    env.exists.return_value = False
    assert env.client.get(_base()).status_code == 404


def test_list_never_contains_statement_hints_or_solution(env):
    env.job.return_value = job_row("done")
    env.exercises.return_value = [exercise_row(position=1), exercise_row(position=2)]

    body = env.client.get(_base()).json()

    assert body["status"] == "done" and len(body["exercises"]) == 2
    for item in body["exercises"]:
        assert set(item) == {"id", "section_id", "position", "title", "covers", "duration_minutes"}
    raw = env.client.get(_base()).text
    assert "SOLUTION-SECRETE" not in raw and "statement" not in raw and "hints" not in raw


def test_list_exposes_each_exercise_duration(env):
    env.job.return_value = job_row("done")
    env.exercises.return_value = [exercise_row(position=1, duration_minutes=7), exercise_row(position=2)]

    items = env.client.get(_base()).json()["exercises"]

    assert [i["duration_minutes"] for i in items] == [7, 10]


def test_list_done_with_zero_exercise_is_valid(env):
    env.job.return_value = job_row("done")
    assert env.client.get(_base()).json() == {"status": "done", "error_message": None, "exercises": []}


def test_list_reports_failure_message_and_hides_it_otherwise(env):
    env.job.return_value = job_row("failed", error_message="Quota du jour")
    assert env.client.get(_base()).json()["error_message"] == "Quota du jour"
    env.job.return_value = job_row("pending", error_message="Nouvelle tentative en cours.")
    body = env.client.get(_base()).json()
    assert body["status"] == "pending" and body["error_message"] is None
    env.exercises.assert_not_awaited()


def test_detail_returns_statement_hints_duration_without_solution(env):
    ex = exercise_row()
    env.one.return_value = ex
    res = env.client.get(f"{_base()}/{ex.id}")
    assert res.status_code == 200
    body = res.json()
    assert body["statement"][0]["type"] == "text" and body["hints"] == ["a", "b", "c"]
    assert body["duration_minutes"] == 10
    assert "expected_solution" not in body and "SOLUTION-SECRETE" not in res.text


def test_detail_of_another_sessions_exercise_is_404(env):
    env.one.return_value = None  # le repository filtre sur (session, exercice)
    ex_id, session_id = uuid.uuid4(), uuid.uuid4()
    assert env.client.get(f"{_base(session_id)}/{ex_id}").status_code == 404
    assert env.one.await_args.args[1:] == (session_id, ex_id)


# --- correction ---------------------------------------------------------------------------------

def _answer(env, answer="R = 30 ohms", ex_id=None):
    return env.client.post(f"{_base()}/{ex_id or uuid.uuid4()}/answer", json={"answer": answer})


def test_answer_returns_feedback_without_score(env):
    res = _answer(env)
    assert res.status_code == 200
    assert res.json() == {
        "feedback": "Bien joué.", "correct_points": ["Série"], "missing_points": ["Unité"],
        "expected_approach": "Additionner les résistances.",
    }
    assert "score" not in res.text and "SOLUTION-SECRETE" not in res.text


def test_answer_reads_statement_and_solution_from_database(env):
    env.one.return_value = exercise_row(expected_solution="LA-VRAIE-SOLUTION")
    env.client.post(f"{_base()}/{uuid.uuid4()}/answer", json={"answer": "x", "expected_solution": "piraté"})
    prompt = env.gemini.format_structured.await_args.kwargs["raw_answer"]
    assert "LA-VRAIE-SOLUTION" in prompt and "piraté" not in prompt


def test_answer_unknown_exercise_is_404(env):
    env.one.return_value = None
    assert _answer(env).status_code == 404
    env.gemini.format_structured.assert_not_awaited()


@pytest.mark.parametrize("answer", ["", "   ", "x" * 4001], ids=["empty", "blank", "too-long"])
def test_answer_rejects_empty_or_too_long(env, answer):
    assert _answer(env, answer).status_code == 422
    env.gemini.format_structured.assert_not_awaited()


def test_answer_is_rate_limited(env):
    env.app.dependency_overrides[routes.get_settings] = lambda: SimpleNamespace(
        exercise_generate_rate_limit_per_minute=2, exercise_answer_rate_limit_per_minute=2,
        exercise_answer_max_length=4000,
    )
    assert [_answer(env).status_code for _ in range(3)] == [200, 200, 429]


def test_answer_invalid_gemini_output_is_explicit_retry_error(env):
    env.gemini.format_structured.side_effect = GeminiInvalidResponseError("x")
    res = _answer(env)
    assert res.status_code == 502 and "Réessayez" in res.json()["detail"]


def test_answer_gemini_429_keeps_retry_after_and_invites_to_retry(env):
    env.gemini.format_structured.side_effect = GeminiQuotaExceededError("429", retry_after_seconds=12)
    res = _answer(env)
    assert res.status_code == 429 and "Réessayez" in res.json()["detail"]
    assert res.headers["Retry-After"] == "12"
