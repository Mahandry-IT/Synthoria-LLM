import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import podcast_routes
from app.api.routes import _maybe_enqueue_podcast
from app.api.schemas import CourseFromPlanRequest, CourseGenerationRequest
from app.core.config import Settings, get_settings

COURSE = {
    "mode": "question_only",
    "format": "focused_answer",
    "meta": {"title": "T", "subject": "S", "language": "fr", "generated_at": "2026-09-20T10:00:00Z"},
    "sources": [],
    "answer": {
        "quoi": "Définition.", "pourquoi": "Raison.", "comment": "Méthode.",
        "worked_example": {"statement": "s", "steps": [], "result": "r"}, "key_points": [],
    },
    "summary": "Résumé.",
    "next_steps": [],
}
EMPTY_COURSE = {**COURSE, "summary": "", "answer": {
    "quoi": "", "pourquoi": "", "comment": "",
    "worked_example": {"statement": "", "steps": [], "result": ""}, "key_points": []}}


class FakeSessionFactory:
    def __call__(self):
        return self

    async def __aenter__(self):
        return object()

    async def __aexit__(self, *exc):
        return False


def make_job(status: str = "done", **overrides):
    now = datetime.now(timezone.utc)
    fields = dict(
        id=uuid.uuid4(), course_session_id=uuid.uuid4(), status=status, stage=status, progress=100 if status == "done" else 10,
        error_message=None, duration_seconds=12.5, script=None, created_at=now, updated_at=now,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.fixture
def env(tmp_path, monkeypatch):
    settings = Settings(gemini_api_key="k", podcast_storage_dir=str(tmp_path / "podcasts"),
                        podcast_generate_rate_limit_per_minute=3)
    app = FastAPI()
    app.include_router(podcast_routes.router)
    app.state.db_session_factory = FakeSessionFactory()
    app.dependency_overrides[get_settings] = lambda: settings
    monkeypatch.setattr(podcast_routes.course_session_repository, "get_by_id",
                        AsyncMock(return_value=SimpleNamespace(gemini_response=COURSE)))
    return SimpleNamespace(client=TestClient(app), app=app, settings=settings, tmp=tmp_path, monkeypatch=monkeypatch)


def patch_job(env, job):
    env.monkeypatch.setattr(podcast_routes.podcast_job_repository, "get_by_id", AsyncMock(return_value=job))


def write_audio(env, job, content: bytes = b"0123456789ABCDEF") -> Path:
    directory = Path(env.settings.podcast_storage_dir) / str(job.id)
    directory.mkdir(parents=True)
    (directory / "podcast.mp3").write_bytes(content)
    (directory / "transcript.vtt").write_text("WEBVTT\n\n", encoding="utf-8")
    return directory


# ─── POST /podcasts/generate/{session_id} ────────────────────


def test_generate_returns_202_with_job(env):
    job = make_job("pending")
    enqueue = AsyncMock(return_value=(job, True))
    env.monkeypatch.setattr(podcast_routes, "enqueue_podcast_job", enqueue)

    res = env.client.post(f"/podcasts/generate/{uuid.uuid4()}")

    assert res.status_code == 202
    assert res.json() == {"job_id": str(job.id), "status": "pending"}
    assert enqueue.await_args.args[2] is None


def test_generate_passes_request_body(env):
    enqueue = AsyncMock(return_value=(make_job("pending"), True))
    env.monkeypatch.setattr(podcast_routes, "enqueue_podcast_job", enqueue)

    res = env.client.post(f"/podcasts/generate/{uuid.uuid4()}", json={"style": "concise", "target_minutes": 5, "force": True})

    assert res.status_code == 202
    body = enqueue.await_args.args[2]
    assert (body.style, body.target_minutes, body.force) == ("concise", 5, True)


@pytest.mark.parametrize("payload", [{"target_minutes": 2}, {"target_minutes": 61}, {"style": "rap"}])
def test_generate_validates_body_bounds(env, payload):
    assert env.client.post(f"/podcasts/generate/{uuid.uuid4()}", json=payload).status_code == 422


def test_generate_rejects_non_uuid_session(env):
    assert env.client.post("/podcasts/generate/pas-un-uuid").status_code == 422


def test_generate_404_for_unknown_session(env):
    env.monkeypatch.setattr(podcast_routes.course_session_repository, "get_by_id", AsyncMock(return_value=None))
    assert env.client.post(f"/podcasts/generate/{uuid.uuid4()}").status_code == 404


def test_generate_422_when_course_has_no_usable_content(env):
    env.monkeypatch.setattr(podcast_routes.course_session_repository, "get_by_id",
                            AsyncMock(return_value=SimpleNamespace(gemini_response=EMPTY_COURSE)))
    res = env.client.post(f"/podcasts/generate/{uuid.uuid4()}")
    assert res.status_code == 422 and "exploitable" in res.json()["detail"]


def test_generate_503_when_feature_disabled(env):
    env.app.dependency_overrides[get_settings] = lambda: Settings(gemini_api_key="k", podcast_enabled=False)
    assert env.client.post(f"/podcasts/generate/{uuid.uuid4()}").status_code == 503


def test_generate_is_rate_limited(env):
    env.monkeypatch.setattr(podcast_routes, "enqueue_podcast_job", AsyncMock(return_value=(make_job("pending"), True)))
    codes = [env.client.post(f"/podcasts/generate/{uuid.uuid4()}").status_code for _ in range(5)]
    assert codes == [202, 202, 202, 429, 429]


# ─── État, audio, transcript, script ─────────────────────────


def test_job_status_and_404(env):
    job = make_job("synthesizing")
    patch_job(env, job)
    body = env.client.get(f"/podcasts/jobs/{job.id}").json()
    assert body["status"] == "synthesizing" and body["progress"] == 10 and body["job_id"] == str(job.id)

    patch_job(env, None)
    assert env.client.get(f"/podcasts/jobs/{uuid.uuid4()}").status_code == 404


def test_audio_409_when_not_done(env):
    job = make_job("mixing")
    patch_job(env, job)
    assert env.client.get(f"/podcasts/{job.id}/audio").status_code == 409


def test_audio_full_file_advertises_ranges(env):
    job = make_job()
    patch_job(env, job)
    write_audio(env, job)

    res = env.client.get(f"/podcasts/{job.id}/audio")

    assert res.status_code == 200 and res.content == b"0123456789ABCDEF"
    assert res.headers["content-type"] == "audio/mpeg" and res.headers["accept-ranges"] == "bytes"


@pytest.mark.parametrize(
    ("header", "expected", "content_range"),
    [
        ("bytes=0-3", b"0123", "bytes 0-3/16"),
        ("bytes=10-", b"ABCDEF", "bytes 10-15/16"),
        ("bytes=-4", b"CDEF", "bytes 12-15/16"),
        ("bytes=8-999", b"89ABCDEF", "bytes 8-15/16"),
    ],
)
def test_audio_range_requests(env, header, expected, content_range):
    job = make_job()
    patch_job(env, job)
    write_audio(env, job)

    res = env.client.get(f"/podcasts/{job.id}/audio", headers={"Range": header})

    assert res.status_code == 206 and res.content == expected
    assert res.headers["content-range"] == content_range and res.headers["content-length"] == str(len(expected))


@pytest.mark.parametrize("header", ["bytes=20-30", "bytes=5-2", "items=0-1", "bytes=-", "bytes=-0"])
def test_audio_invalid_range_is_416(env, header):
    job = make_job()
    patch_job(env, job)
    write_audio(env, job)

    res = env.client.get(f"/podcasts/{job.id}/audio", headers={"Range": header})

    assert res.status_code == 416 and res.headers["content-range"] == "bytes */16"


def test_audio_410_when_file_was_purged(env):
    job = make_job()
    patch_job(env, job)
    assert env.client.get(f"/podcasts/{job.id}/audio").status_code == 410


def test_transcript_served_as_vtt(env):
    job = make_job()
    patch_job(env, job)
    write_audio(env, job)

    res = env.client.get(f"/podcasts/{job.id}/transcript")

    assert res.status_code == 200 and res.text.startswith("WEBVTT")
    assert res.headers["content-type"].startswith("text/vtt")


def test_script_409_then_200(env):
    job = make_job("scripting")
    patch_job(env, job)
    assert env.client.get(f"/podcasts/{job.id}/script").status_code == 409

    ready = make_job("synthesizing", script={
        "title": "T", "intro_turns": [{"speaker": "HOST", "text": "Bonjour."}],
        "segments": [], "outro_turns": [{"speaker": "EXPERT", "text": "Au revoir."}],
    })
    patch_job(env, ready)
    assert env.client.get(f"/podcasts/{ready.id}/script").json()["title"] == "T"


def test_session_history_lists_jobs(env):
    jobs = [make_job("done"), make_job("failed", error_message="boom")]
    env.monkeypatch.setattr(podcast_routes.podcast_job_repository, "list_by_session", AsyncMock(return_value=jobs))

    res = env.client.get(f"/courses/history/{uuid.uuid4()}/podcasts")

    assert [j["status"] for j in res.json()["data"]] == ["done", "failed"]
    assert res.json()["data"][1]["error_message"] == "boom"


# ─── Déclenchement automatique après le cours ────────────────


def _request_with_factory():
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(db_session_factory=FakeSessionFactory())))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("requested", "auto", "expected_enqueue"),
    [(None, False, False), (None, True, True), (True, False, True), (False, True, False)],
)
async def test_auto_enqueue_decision(monkeypatch, requested, auto, expected_enqueue):
    from app.api import routes

    job = make_job("pending")
    enqueue = AsyncMock(return_value=(job, True))
    monkeypatch.setattr(routes, "enqueue_podcast_job", enqueue)
    settings = Settings(gemini_api_key="k", podcast_auto_generate=auto)

    result = await _maybe_enqueue_podcast(_request_with_factory(), uuid.uuid4(), requested, settings)

    assert (result == job.id) is expected_enqueue and (enqueue.await_count == 1) is expected_enqueue


@pytest.mark.asyncio
async def test_auto_enqueue_never_raises_and_respects_switches(monkeypatch):
    from app.api import routes

    enqueue = AsyncMock(side_effect=RuntimeError("db down"))
    monkeypatch.setattr(routes, "enqueue_podcast_job", enqueue)
    settings = Settings(gemini_api_key="k")
    request = _request_with_factory()

    assert await _maybe_enqueue_podcast(request, uuid.uuid4(), True, settings) is None       # erreur avalée
    assert await _maybe_enqueue_podcast(request, None, True, settings) is None                # pas de session persistée
    off = Settings(gemini_api_key="k", podcast_enabled=False)
    assert await _maybe_enqueue_podcast(request, uuid.uuid4(), True, off) is None             # fonctionnalité coupée
    assert enqueue.await_count == 1


def test_course_requests_accept_optional_generate_podcast():
    assert CourseGenerationRequest().generate_podcast is None
    assert CourseGenerationRequest(generate_podcast=True).generate_podcast is True
    plan = {"plan_id": str(uuid.uuid4()), "sections": [{"type": "development", "title": "A", "order": 1}]}
    assert CourseFromPlanRequest.model_validate(plan).generate_podcast is None
    assert CourseFromPlanRequest.model_validate({**plan, "generate_podcast": False}).generate_podcast is False


# ─── GET /podcasts (récents, dashboard) ──────────────────────


def test_recent_podcasts_titles_fall_back_from_script_to_course_to_question(env):
    def course(meta_title=None):
        meta = {"title": meta_title} if meta_title else {}
        return SimpleNamespace(gemini_response={"meta": meta}, question="Une très longue question ?")

    rows = [
        (make_job("done", script={"title": "Titre du script"}), course("Titre du cours")),
        (make_job("done"), course("Titre du cours")),
        (make_job("pending"), course()),
    ]
    mock = AsyncMock(return_value=rows)
    env.monkeypatch.setattr(podcast_routes.podcast_job_repository, "list_recent", mock)

    res = env.client.get("/podcasts?limit=3")

    assert res.status_code == 200
    assert [p["title"] for p in res.json()["data"]] == ["Titre du script", "Titre du cours", "Une très longue question ?"]
    assert res.json()["data"][0]["status"] == "done"
    assert mock.call_args.args[1] == 3


def test_recent_podcasts_default_limit_is_3(env):
    mock = AsyncMock(return_value=[])
    env.monkeypatch.setattr(podcast_routes.podcast_job_repository, "list_recent", mock)

    assert env.client.get("/podcasts").json() == {"data": []}
    assert mock.call_args.args[1] == 3


@pytest.mark.parametrize("limit", [0, 21, "abc"])
def test_recent_podcasts_validates_limit(env, limit):
    assert env.client.get(f"/podcasts?limit={limit}").status_code == 422
