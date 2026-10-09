import random
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.core.exceptions import GeminiQuotaExceededError
from app.services import quiz_bank

NOW = datetime(2026, 10, 9, tzinfo=timezone.utc)


def _q(text: str, difficulty: str = "normale", correct: list[int] | None = None) -> dict:
    return {
        "question": text,
        "options": ["A", "B", "C"],
        "correct_option_indices": correct if correct is not None else [0],
        "difficulty": difficulty,
        "points": 1.0,
        "explanation": f"Explication {text}",
        "explanation_per_choice": ["a", "b", "c"],
        "section_refs": [],
        "time_limit_seconds": 45,
    }


def _entry(payload: dict, served: int = 0, last: datetime | None = None) -> SimpleNamespace:
    return SimpleNamespace(id=uuid.uuid4(), payload=payload, times_served=served, last_served_at=last)


# ─── Logique pure ────────────────────────────────────────────────────────────


def test_original_quiz_normalizes_legacy_fields_and_drops_invalid_questions():
    legacy = {"question": "Q", "options": ["A", "B"], "correct_option_index": 1}
    no_answer = {"question": "R", "options": ["A", "B"]}
    broken = {"question": "S", "options": ["A"]}

    quiz = quiz_bank.original_quiz({"quiz": [legacy, no_answer, broken]})

    assert [q["question"] for q in quiz] == ["Q"]
    assert quiz[0]["correct_option_indices"] == [1]


def test_select_prefers_least_served_and_keeps_difficulty_distribution():
    fresh_easy = _entry(_q("f1", "facile"))
    served_easy = _entry(_q("f2", "facile"), served=3, last=NOW)
    fresh_hard = [_entry(_q(f"d{i}", "difficile")) for i in range(2)]
    served_hard = _entry(_q("d9", "difficile"), served=1, last=NOW)
    bank = [served_easy, fresh_easy, served_hard, *fresh_hard]

    selected = quiz_bank.select_questions(bank, 3, Counter({"facile": 1, "difficile": 2}), random.Random(1))

    assert {q.payload["question"] for q in selected} == {"f1", "d0", "d1"}
    assert selected[0].payload["difficulty"] == "facile"  # ordre facile → difficile


def test_select_breaks_ties_on_oldest_service_then_fills_missing_difficulty():
    old = _entry(_q("old"), served=1, last=NOW - timedelta(days=2))
    recent = _entry(_q("recent"), served=1, last=NOW)
    hard = _entry(_q("hard", "difficile"), served=5, last=NOW)

    # 2 « facile » demandées, aucune disponible : complétées par les moins servies.
    selected = quiz_bank.select_questions([recent, hard, old], 2, Counter({"facile": 2}), random.Random(0))

    assert {q.payload["question"] for q in selected} == {"old", "recent"}


def test_select_random_tie_break_varies_with_the_seed():
    bank = [_entry(_q(f"q{i}")) for i in range(10)]
    draws = {
        tuple(q.payload["question"] for q in quiz_bank.select_questions(bank, 3, Counter({"normale": 3}), random.Random(seed)))
        for seed in range(5)
    }
    assert len(draws) > 1


def test_public_question_hides_answers_and_explanations():
    public = quiz_bank.public_question(_q("Q"), 4.5)

    assert public["correct_option_indices"] == []
    assert public["explanation"] == "" and public["explanation_per_choice"] == []
    assert public["points"] == 4.5 and public["options"] == ["A", "B", "C"]


def test_compute_points_totals_twenty():
    payloads = [_q("a", "facile"), _q("b", "normale"), _q("c", "difficile"), _q("d", "difficile")]
    assert sum(quiz_bank.compute_points(payloads)) == pytest.approx(20.0)


def test_grade_requires_exact_set_equality():
    payloads = [_q("a", correct=[0, 2]), _q("b", correct=[1]), _q("c", correct=[2]), _q("d", correct=[0])]
    points = quiz_bank.compute_points(payloads)

    score, results = quiz_bank.grade(payloads, [[2, 0], [1, 2], [], [0]])

    assert [r["is_correct"] for r in results] == [True, False, False, True]
    assert score == pytest.approx(points[0] + points[3])
    assert results[1]["correct_option_indices"] == [1] and results[1]["points_earned"] == 0.0
    assert results[0]["explanation"] == "Explication a"


@pytest.mark.parametrize(
    "answers, partial",
    [([[0]], False), ([[0], [0], [0]], True), ([[0, 0], [1]], False), ([[3], [1]], False)],
)
def test_validate_answers_rejects_inconsistent_answers(answers, partial):
    with pytest.raises(quiz_bank.InvalidQuizAnswers):
        quiz_bank.validate_answers(answers, [_q("a"), _q("b")], partial=partial)


def test_validate_answers_accepts_partial_answers_for_an_aborted_attempt():
    quiz_bank.validate_answers([[1]], [_q("a"), _q("b")], partial=True)


def test_needs_refill_when_fewer_never_served_than_quiz_size():
    bank = [_entry(_q("a")), _entry(_q("b"), served=1)]
    assert quiz_bank.needs_refill(bank, 2) is True
    assert quiz_bank.needs_refill(bank, 1) is False


def test_parse_refill_keeps_valid_new_questions_only():
    def gemini_q(text, correct=None):
        return {
            "question": text, "choices": ["A", "B"], "correct_indices": correct if correct is not None else [1],
            "difficulty": "difficile", "explanation": "x", "requires_calculation": False,
        }

    structured = {"questions": [gemini_q("Existante ?"), gemini_q("Nouvelle"), gemini_q("nouvelle !"), gemini_q("Vide", [])]}

    payloads = quiz_bank.parse_refill(structured, [_q("existante")])

    assert [p["question"] for p in payloads] == ["Nouvelle"]
    assert payloads[0]["options"] == ["A", "B"] and payloads[0]["correct_option_indices"] == [1]


def test_refill_prompt_contains_course_rules_and_existing_questions():
    course = {"meta": {"title": "Cours"}, "quiz": [_q("Ancienne", "difficile"), _q("Autre", "facile")]}

    prompt = quiz_bank.build_refill_prompt(course, course["quiz"], 2, Settings(gemini_api_key="k"))

    assert "exactement 2 NOUVELLES questions" in prompt
    assert "1 difficile" in prompt and "1 facile" in prompt
    assert "- Ancienne" in prompt


# ─── Tentatives (dépôt en mémoire) ───────────────────────────────────────────


class FakeDb:
    def __init__(self) -> None:
        self.commits = 0

    async def commit(self) -> None:
        self.commits += 1


@pytest.fixture
def store(monkeypatch):
    sid = uuid.uuid4()
    quiz = [_q("q1", "facile"), _q("q2", "difficile")]
    state = SimpleNamespace(
        sid=sid, row=SimpleNamespace(id=sid, gemini_response={"quiz": quiz}), bank=[], attempts={},
    )
    repo = quiz_bank.quiz_repository

    async def lock_session(db, session_id):
        return state.row if session_id == sid else None

    async def list_bank(db, session_id):
        return list(state.bank)

    def add_questions(db, session_id, payloads, batch):
        rows = [SimpleNamespace(id=uuid.uuid4(), payload=p, batch=batch, times_served=0, last_served_at=None) for p in payloads]
        state.bank.extend(rows)
        return rows

    def create_attempt(db, session_id, question_ids, max_score, now):
        attempt = SimpleNamespace(
            id=uuid.uuid4(), session_id=session_id, question_ids=[str(q) for q in question_ids], status="in_progress",
            answers=None, score=None, abort_reason=None, finished_at=None,
        )
        state.attempts[attempt.id] = attempt
        return attempt

    async def get_attempt_for_update(db, session_id, attempt_id):
        attempt = state.attempts.get(attempt_id)
        return attempt if attempt and attempt.session_id == session_id else None

    async def get_questions(db, ids):
        return {q.id: q for q in state.bank if q.id in ids}

    async def max_batch(db, session_id):
        return max((q.batch for q in state.bank), default=-1)

    for name, fn in {
        "lock_session": lock_session, "list_bank": list_bank, "add_questions": add_questions,
        "create_attempt": create_attempt, "get_attempt_for_update": get_attempt_for_update,
        "get_questions": get_questions, "max_batch": max_batch,
    }.items():
        monkeypatch.setattr(repo, name, fn)
    monkeypatch.setattr(quiz_bank.course_session_repository, "get_by_id", AsyncMock(side_effect=lambda db, i: state.row if i == sid else None))
    return state


@pytest.mark.asyncio
async def test_first_attempt_seeds_the_bank_with_the_original_quiz(store):
    db = FakeDb()

    started = await quiz_bank.start_attempt(db, store.sid, rng=random.Random(0), now=NOW)

    assert [q.batch for q in store.bank] == [0, 0]
    assert {q["question"] for q in started.questions} == {"q1", "q2"}
    assert all(q["correct_option_indices"] == [] for q in started.questions)
    assert sum(q["points"] for q in started.questions) == pytest.approx(20.0)
    assert all(q.times_served == 1 and q.last_served_at == NOW for q in store.bank)
    assert started.refill_needed is True  # plus aucune question jamais servie
    assert db.commits == 1


@pytest.mark.asyncio
async def test_start_attempt_unknown_session_or_course_without_quiz(store):
    with pytest.raises(quiz_bank.QuizSessionNotFound):
        await quiz_bank.start_attempt(FakeDb(), uuid.uuid4())
    store.row.gemini_response = {"quiz": []}
    with pytest.raises(quiz_bank.QuizUnavailable):
        await quiz_bank.start_attempt(FakeDb(), store.sid)


@pytest.mark.asyncio
async def test_new_attempt_draws_fresh_questions_after_a_refill(store):
    await quiz_bank.start_attempt(FakeDb(), store.sid, now=NOW)
    quiz_bank.quiz_repository.add_questions(None, store.sid, [_q("n1", "facile"), _q("n2", "difficile")], batch=1)

    started = await quiz_bank.start_attempt(FakeDb(), store.sid, now=NOW + timedelta(minutes=1))

    assert {q["question"] for q in started.questions} == {"n1", "n2"}


@pytest.mark.asyncio
async def test_submit_grades_out_of_twenty_then_refuses_a_second_submission(store):
    started = await quiz_bank.start_attempt(FakeDb(), store.sid, rng=random.Random(0), now=NOW)
    # q1 (facile) puis q2 (difficile) ; bonne réponse = [0] pour les deux.

    submitted = await quiz_bank.submit_attempt(FakeDb(), store.sid, started.attempt_id, [[0], [0]], now=NOW)

    assert submitted.status == "completed" and submitted.score == pytest.approx(20.0)
    assert submitted.max_score == 20.0 and len(submitted.results) == 2
    assert store.attempts[started.attempt_id].score == pytest.approx(20.0)
    with pytest.raises(quiz_bank.QuizAttemptAlreadySubmitted):
        await quiz_bank.submit_attempt(FakeDb(), store.sid, started.attempt_id, [[0], [0]])


@pytest.mark.asyncio
async def test_partial_score_counts_only_exactly_correct_questions(store):
    started = await quiz_bank.start_attempt(FakeDb(), store.sid, rng=random.Random(0), now=NOW)
    hard_points = started.questions[1]["points"]

    submitted = await quiz_bank.submit_attempt(FakeDb(), store.sid, started.attempt_id, [[1], [0]])

    assert submitted.score == pytest.approx(hard_points)


@pytest.mark.asyncio
async def test_aborted_attempt_is_recorded_without_score(store):
    started = await quiz_bank.start_attempt(FakeDb(), store.sid, now=NOW)

    submitted = await quiz_bank.submit_attempt(
        FakeDb(), store.sid, started.attempt_id, [[0]], aborted=True, abort_reason="fullscreen_exit"
    )

    attempt = store.attempts[started.attempt_id]
    assert submitted.status == "aborted" and submitted.score is None and submitted.results == []
    assert attempt.status == "aborted" and attempt.abort_reason == "fullscreen_exit" and attempt.score is None


@pytest.mark.asyncio
async def test_submit_unknown_attempt_or_invalid_answers(store):
    with pytest.raises(quiz_bank.QuizAttemptNotFound):
        await quiz_bank.submit_attempt(FakeDb(), store.sid, uuid.uuid4(), [])
    started = await quiz_bank.start_attempt(FakeDb(), store.sid, now=NOW)
    with pytest.raises(quiz_bank.QuizAttemptNotFound):  # tentative d'un autre cours
        await quiz_bank.submit_attempt(FakeDb(), uuid.uuid4(), started.attempt_id, [[0], [0]])
    with pytest.raises(quiz_bank.InvalidQuizAnswers):
        await quiz_bank.submit_attempt(FakeDb(), store.sid, started.attempt_id, [[0]])
    assert store.attempts[started.attempt_id].status == "in_progress"


# ─── Recharge ────────────────────────────────────────────────────────────────


def _factory():
    @asynccontextmanager
    async def factory():
        yield FakeDb()

    return factory


@pytest.mark.asyncio
async def test_refill_adds_a_new_batch_with_one_gemini_call(store):
    await quiz_bank.start_attempt(FakeDb(), store.sid, now=NOW)
    gemini = AsyncMock()
    gemini.format_structured.return_value = {
        "questions": [
            {"question": f"Nouvelle {i}", "choices": ["A", "B"], "correct_indices": [0], "difficulty": "normale",
             "explanation": "x", "requires_calculation": False}
            for i in range(2)
        ]
    }
    in_flight = {store.sid}

    await quiz_bank.refill_bank(store.sid, _factory(), gemini, Settings(gemini_api_key="k"), in_flight)

    gemini.format_structured.assert_awaited_once()
    assert [q.batch for q in store.bank] == [0, 0, 1, 1]
    assert in_flight == set()


@pytest.mark.asyncio
async def test_refill_failure_is_swallowed_and_draw_falls_back_to_least_served(store):
    await quiz_bank.start_attempt(FakeDb(), store.sid, now=NOW)
    gemini = AsyncMock()
    gemini.format_structured.side_effect = GeminiQuotaExceededError("quota")
    in_flight = {store.sid}

    await quiz_bank.refill_bank(store.sid, _factory(), gemini, Settings(gemini_api_key="k"), in_flight)

    assert len(store.bank) == 2 and in_flight == set()
    again = await quiz_bank.start_attempt(FakeDb(), store.sid, now=NOW + timedelta(minutes=1))
    assert {q["question"] for q in again.questions} == {"q1", "q2"}  # répétition assumée


@pytest.mark.asyncio
async def test_refill_skips_gemini_when_bank_still_has_enough_fresh_questions(store):
    quiz_bank.quiz_repository.add_questions(None, store.sid, [_q("a"), _q("b"), _q("c")], batch=0)
    gemini = AsyncMock()

    await quiz_bank.refill_bank(store.sid, _factory(), gemini, Settings(gemini_api_key="k"), set())

    gemini.format_structured.assert_not_called()
