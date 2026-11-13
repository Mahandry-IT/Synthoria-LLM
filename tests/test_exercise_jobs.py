import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.core.exceptions import (
    GeminiDailyQuotaExceededError,
    GeminiInvalidResponseError,
    GeminiQuotaExceededError,
    GeminiUnavailableError,
)
from app.services import exercise_jobs as svc
from tests.exercise_fixtures import course, draft, job_row

SETTINGS = Settings(gemini_api_key="k", exercise_max_attempts=3, exercise_retry_delay_seconds=30)


class Factory:
    def __call__(self):
        return self

    async def __aenter__(self):
        self.db = AsyncMock()
        return self.db

    async def __aexit__(self, *exc):
        return False


@pytest.fixture
def repos(monkeypatch):
    jobs = SimpleNamespace(
        get_by_session=AsyncMock(), create_if_absent=AsyncMock(return_value=True), restart_failed=AsyncMock(),
        mark_done=AsyncMock(return_value=True), mark_failed=AsyncMock(), requeue=AsyncMock(),
    )
    monkeypatch.setattr(svc, "jobs_repo", jobs)
    monkeypatch.setattr(svc, "exercise_repository", SimpleNamespace(replace_all=AsyncMock()))
    monkeypatch.setattr(
        svc.course_session_repository, "get_by_id",
        AsyncMock(return_value=SimpleNamespace(gemini_response=course("standard"))),
    )
    return jobs


# --- mise en file idempotente ---------------------------------------------------------------

@pytest.mark.asyncio
async def test_enqueue_creates_job_when_absent(repos):
    repos.get_by_session.side_effect = [None, job_row("pending")]
    job = await svc.enqueue_exercise_job(Factory(), uuid.uuid4(), course(), SETTINGS)
    repos.create_if_absent.assert_awaited_once()
    assert job.status == "pending"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["pending", "generating", "done"])
async def test_enqueue_never_touches_an_existing_non_failed_job(repos, status):
    repos.get_by_session.return_value = job_row(status)
    job = await svc.enqueue_exercise_job(Factory(), uuid.uuid4(), course(), SETTINGS)
    assert job.status == status
    repos.create_if_absent.assert_not_awaited()
    repos.restart_failed.assert_not_awaited()


@pytest.mark.asyncio
async def test_enqueue_restarts_only_failed(repos):
    repos.get_by_session.side_effect = [job_row("failed"), job_row("pending")]
    job = await svc.enqueue_exercise_job(Factory(), uuid.uuid4(), course(), SETTINGS)
    repos.restart_failed.assert_awaited_once()
    assert job.status == "pending"


@pytest.mark.asyncio
async def test_losing_concurrent_launch_does_not_insert_nor_restart(repos):
    """Les deux lancements voient `absent` ; l'INSERT ON CONFLICT du second ne crée rien (False) : il ne
    relance rien (ni restart_failed) et renvoie la ligne du premier, sans second job ni appel Gemini."""
    shared = job_row("pending")
    repos.get_by_session.side_effect = [None, shared, None, shared]
    repos.create_if_absent.side_effect = [True, False]
    gemini = _gemini({"exercises": []})

    first = await svc.enqueue_exercise_job(Factory(), shared.session_id, course(), SETTINGS)
    second = await svc.enqueue_exercise_job(Factory(), shared.session_id, course(), SETTINGS)

    assert first is second is shared
    assert repos.create_if_absent.await_count == 2  # une tentative d'INSERT par lancement, une seule ligne créée
    repos.restart_failed.assert_not_awaited()
    gemini.format_structured.assert_not_awaited()  # la mise en file ne génère jamais : le worker s'en charge


# --- exécution --------------------------------------------------------------------------------

def _gemini(result=None, error=None) -> AsyncMock:
    client = AsyncMock()
    if error:
        client.format_structured.side_effect = error
    else:
        client.format_structured.return_value = result
    return client


async def _run(job, client, **kw):
    return await svc.run_exercise_job(job, session_factory=Factory(), gemini_client=client, settings=SETTINGS, **kw)


@pytest.mark.asyncio
async def test_success_saves_exercises_and_done_in_same_transaction(repos):
    client = _gemini({"exercises": [draft()]})
    assert await _run(job_row("generating", attempts=1), client) == "done"
    client.format_structured.assert_awaited_once()
    svc.exercise_repository.replace_all.assert_awaited_once()
    repos.mark_done.assert_awaited_once()


@pytest.mark.asyncio
async def test_done_is_marked_before_exercises_are_written_then_one_commit(repos):
    order = []
    repos.mark_done.side_effect = lambda *a, **k: order.append("mark_done") or True
    svc.exercise_repository.replace_all.side_effect = lambda *a, **k: order.append("replace_all")
    factory = Factory()
    result = await svc.run_exercise_job(
        job_row("generating", attempts=1), session_factory=factory, gemini_client=_gemini({"exercises": [draft()]}),
        settings=SETTINGS,
    )
    assert result == "done" and order == ["mark_done", "replace_all"]
    factory.db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_lost_job_writes_no_exercise(repos):
    """Le job n'est plus à nous (libéré comme périmé) : `done` est refusé et aucun exercice n'est écrit."""
    repos.mark_done.return_value = False
    assert await _run(job_row("generating", attempts=1), _gemini({"exercises": [draft()]})) == "lost"
    svc.exercise_repository.replace_all.assert_not_awaited()


@pytest.mark.asyncio
async def test_zero_exercise_is_a_valid_done(repos):
    assert await _run(job_row("generating", attempts=1), _gemini({"exercises": []})) == "done"
    assert svc.exercise_repository.replace_all.await_args.args[2] == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [GeminiUnavailableError("x"), GeminiInvalidResponseError("x")],
)
async def test_gemini_failure_requeues_without_partial_exercises(repos, error):
    assert await _run(job_row("generating", attempts=1), _gemini(error=error)) == "requeued"
    repos.requeue.assert_awaited_once()
    assert repos.requeue.await_args.kwargs["delay_seconds"] >= 30
    svc.exercise_repository.replace_all.assert_not_awaited()
    repos.mark_done.assert_not_awaited()


@pytest.mark.asyncio
async def test_minute_429_refunds_the_attempt_and_waits(repos):
    # Même au plafond de tentatives : un 429 « minute » n'est pas la faute du job.
    result = await _run(
        job_row("generating", attempts=3), _gemini(error=GeminiQuotaExceededError("429", retry_after_seconds=45))
    )
    assert result == "requeued"
    kwargs = repos.requeue.await_args.kwargs
    assert kwargs["refund_attempt"] is True and kwargs["delay_seconds"] == 45
    repos.mark_failed.assert_not_awaited()
    svc.exercise_repository.replace_all.assert_not_awaited()


@pytest.mark.asyncio
async def test_other_failures_do_not_refund_the_attempt(repos):
    await _run(job_row("generating", attempts=1), _gemini(error=GeminiUnavailableError("x")))
    assert not repos.requeue.await_args.kwargs.get("refund_attempt")


@pytest.mark.asyncio
async def test_429_waits_at_least_google_retry_delay(repos):
    await _run(job_row("generating", attempts=1), _gemini(error=GeminiQuotaExceededError("429", retry_after_seconds=90)))
    assert repos.requeue.await_args.kwargs["delay_seconds"] == 90


@pytest.mark.asyncio
async def test_attempts_are_bounded(repos):
    result = await _run(job_row("generating", attempts=3), _gemini(error=GeminiUnavailableError("503")))
    assert result == "failed"
    repos.mark_failed.assert_awaited_once()
    repos.requeue.assert_not_awaited()


@pytest.mark.asyncio
async def test_daily_quota_fails_at_once_with_a_clear_message(repos):
    result = await _run(job_row("generating", attempts=1), _gemini(error=GeminiDailyQuotaExceededError("rpd")))
    assert result == "failed"
    assert repos.mark_failed.await_args.args[2] == svc.MSG_QUOTA_DAILY


@pytest.mark.asyncio
async def test_unexpected_error_never_leaves_job_generating(repos):
    assert await _run(job_row("generating", attempts=1), _gemini(error=RuntimeError("boom"))) == "requeued"


@pytest.mark.asyncio
async def test_stop_requeues_without_counting_attempt_or_calling_gemini(repos):
    client = _gemini({"exercises": []})
    assert await _run(job_row("generating", attempts=1), client, should_stop=lambda: True) == "interrupted"
    assert repos.requeue.await_args.kwargs["refund_attempt"] is True
    client.format_structured.assert_not_awaited()


@pytest.mark.asyncio
async def test_deleted_course_fails_the_job(repos):
    svc.course_session_repository.get_by_id.return_value = None
    assert await _run(job_row("generating", attempts=1), _gemini({"exercises": []})) == "failed"
