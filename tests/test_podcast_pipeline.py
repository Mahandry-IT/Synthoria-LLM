import math
import os
import re
import shutil
import struct
import time
import uuid
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.api.schemas import PodcastGenerationRequest
from app.cli import build_parser
from app.core.config import Settings
from app.core.exceptions import TTSInvalidVoiceError, TTSUnavailableError
from app.schemas.podcast import PodcastFrame, PodcastScript
from app.services.podcast import jobs as jobs_module
from app.services.podcast import pipeline
from app.services.podcast.jobs import (
    CourseSessionNotFoundError,
    compute_params_hash,
    enqueue_podcast_job,
    job_dir,
    resolve_params,
)
from app.services.podcast.pipeline import build_spoken_turns, run_podcast_job
from app.workers import podcast_worker

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg absent")

COURSE = {
    "mode": "question_only",
    "format": "focused_answer",
    "meta": {"title": "Green IT", "subject": "S", "language": "fr", "generated_at": "2026-09-20T10:00:00Z"},
    "sources": [],
    "answer": {
        "quoi": "Le Green IT réduit l'impact du numérique.",
        "pourquoi": "Pour limiter les émissions.",
        "comment": "En mesurant puis en optimisant.",
        "worked_example": {"statement": "Un serveur de 200 W.", "steps": [{"id": "1", "content": "Multiplier."}], "result": "4,8 kWh."},
        "key_points": [],
    },
    "summary": "Retenez l'essentiel.",
    "next_steps": ["Écoconception"],
}


def tiny_wav() -> bytes:
    import io

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(22050)
        handle.writeframes(b"".join(struct.pack("<h", int(6000 * math.sin(i / 5))) for i in range(2205)))
    return buffer.getvalue()


class FakeStore:
    """Remplace le repository : état du job en mémoire + historique des appels."""

    def __init__(self, course: dict | None = COURSE, *, attempts: int = 1, params: dict | None = None):
        self.job = SimpleNamespace(
            id=uuid.uuid4(), course_session_id=uuid.uuid4(), status="scripting", stage=None, progress=0,
            params=params or {"style": "conversational", "target_minutes": 3}, script=None, attempts=attempts,
            audio_path=None, duration_seconds=None, error_message=None, voices=None, tts_engine=None,
        )
        self.course_row = SimpleNamespace(gemini_response=course) if course is not None else None
        self.stages: list[tuple[str, int]] = []
        self.requeues: list[dict] = []

    def install(self, monkeypatch) -> None:
        async def get_by_id(db, job_id):
            return self.job

        async def update_stage(db, job_id, *, status, stage, progress):
            self.job.status, self.job.stage, self.job.progress = status, stage, progress
            self.stages.append((status, progress))

        async def save_script(db, job_id, script):
            self.job.script = script

        async def mark_done(db, job_id, *, audio_path, duration_seconds, tts_engine, voices):
            self.job.status, self.job.progress = "done", 100
            self.job.audio_path, self.job.duration_seconds = audio_path, duration_seconds
            self.job.tts_engine, self.job.voices = tts_engine, voices

        async def mark_failed(db, job_id, message):
            self.job.status, self.job.error_message = "failed", message

        async def requeue(db, job_id, *, error_message=None, refund_attempt=False):
            self.job.status = "pending"
            if error_message:
                self.job.error_message = error_message
            self.requeues.append({"error": error_message, "refund": refund_attempt})

        async def course_get(db, session_id):
            return self.course_row

        repo = pipeline.jobs_repo
        for name, fn in [
            ("get_by_id", get_by_id), ("update_stage", update_stage), ("save_script", save_script),
            ("mark_done", mark_done), ("mark_failed", mark_failed), ("requeue", requeue),
        ]:
            monkeypatch.setattr(repo, name, fn)
        monkeypatch.setattr(pipeline.course_session_repository, "get_by_id", course_get)


class FakeSessionFactory:
    def __call__(self):
        return self

    async def __aenter__(self):
        return object()

    async def __aexit__(self, *exc):
        return False


class FakeTTS:
    name = "fake"

    def __init__(self, fail_from_call: int | None = None, error: Exception | None = None):
        self.calls: list[tuple[str, str]] = []
        self._fail_from = fail_from_call
        self._error = error or TTSUnavailableError("moteur coupé")
        self._audio = tiny_wav()

    async def synthesize(self, text, voice):
        if self._fail_from is not None and len(self.calls) >= self._fail_from:
            raise self._error
        self.calls.append((voice, text))
        return self._audio

    async def close(self):
        return None


def fake_gemini() -> AsyncMock:
    async def format_structured(raw_answer, system_instruction, *, response_schema=None):
        if response_schema is PodcastFrame:
            return {
                "title": "Le podcast",
                "intro_turns": [{"speaker": "HOST", "text": "Bienvenue dans ce podcast."}],
                "outro_turns": [{"speaker": "EXPERT", "text": "Merci de votre écoute."}],
            }
        refs = [int(x) for x in re.findall(r'<course_data index="(\d+)"', raw_answer)]
        return {"segments": [
            {"section_ref": r, "title": f"Segment {r}", "turns": [
                {"speaker": "HOST", "text": "Quelle est la valeur de 12 € ?"},
                {"speaker": "EXPERT", "text": "C'est 85 % du total. Deuxième phrase pour le test."},
            ]} for r in refs
        ]}

    client = AsyncMock()
    client.format_structured.side_effect = format_structured
    return client


def settings_for(tmp_path: Path, **overrides) -> Settings:
    return Settings(
        gemini_api_key="k", podcast_storage_dir=str(tmp_path / "podcasts"), podcast_tts_concurrency=2,
        podcast_max_attempts=3, **overrides,
    )


async def fake_assemble(timeline, *, title, output_path, work_dir, bitrate="96k", ffmpeg="ffmpeg"):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(b"ID3fake")
    return output_path


async def run(store, tmp_path, *, gemini=None, tts=None, should_stop=lambda: False, **settings_overrides):
    return await run_podcast_job(
        store.job.id, session_factory=FakeSessionFactory(), gemini_client=gemini or fake_gemini(),
        tts_engine=tts or FakeTTS(), settings=settings_for(tmp_path, **settings_overrides), should_stop=should_stop,
    )


# ─── Pipeline ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pipeline_pending_to_done(monkeypatch, tmp_path):
    store = FakeStore()
    store.install(monkeypatch)
    monkeypatch.setattr(pipeline, "assemble_mp3", fake_assemble)
    tts = FakeTTS()

    outcome = await run(store, tmp_path, tts=tts)

    assert outcome == "done"
    assert store.job.status == "done" and store.job.progress == 100
    assert [s for s, _ in store.stages][:1] == ["scripting"] and ("mixing", 88) in store.stages
    assert store.job.script["title"] == "Le podcast"                       # checkpoint enregistré
    assert store.job.tts_engine == "fake" and set(store.job.voices) == {"HOST", "EXPERT"}
    directory = job_dir(settings_for(tmp_path), store.job.id)
    assert (directory / "podcast.mp3").exists() and (directory / "transcript.vtt").exists()
    assert "WEBVTT" in (directory / "transcript.vtt").read_text(encoding="utf-8")
    assert not (directory / "cache").exists()                              # WAV intermédiaires nettoyés
    voices = {v for v, _ in tts.calls}
    assert voices == {"fr_FR-siwis-medium", "fr_FR-tom-medium"}            # une voix par locuteur
    spoken = " ".join(text for _, text in tts.calls)
    assert "€" not in spoken and "%" not in spoken and "euros" in spoken and "pour cent" in spoken  # normalisé


@needs_ffmpeg
@pytest.mark.asyncio
async def test_pipeline_end_to_end_with_real_ffmpeg(monkeypatch, tmp_path):
    store = FakeStore()
    store.install(monkeypatch)

    outcome = await run(store, tmp_path)

    assert outcome == "done"
    assert Path(store.job.audio_path).stat().st_size > 1000
    assert store.job.duration_seconds and store.job.duration_seconds > 1


@pytest.mark.asyncio
async def test_resume_after_tts_failure_reuses_script_and_cached_audio(monkeypatch, tmp_path):
    store = FakeStore()
    store.install(monkeypatch)
    monkeypatch.setattr(pipeline, "assemble_mp3", fake_assemble)
    gemini = fake_gemini()
    flaky = FakeTTS(fail_from_call=3)

    first = await run(store, tmp_path, gemini=gemini, tts=flaky)

    assert first == "requeued" and store.job.status == "pending"
    assert "TTSUnavailableError" in store.job.error_message
    assert store.job.script is not None
    gemini_calls_after_first = gemini.format_structured.await_count
    done_first = len(flaky.calls)
    assert done_first == 3

    healthy = FakeTTS()
    second = await run(store, tmp_path, gemini=gemini, tts=healthy)

    assert second == "done"
    assert gemini.format_structured.await_count == gemini_calls_after_first      # script non régénéré
    total_unique = len({t for _, t in flaky.calls} | {t for _, t in healthy.calls})
    assert len(healthy.calls) + done_first == total_unique                         # rien resynthétisé


@pytest.mark.asyncio
async def test_exhausted_attempts_mark_job_failed(monkeypatch, tmp_path):
    store = FakeStore(attempts=3)
    store.install(monkeypatch)

    outcome = await run(store, tmp_path, tts=FakeTTS(fail_from_call=0))

    assert outcome == "failed" and store.job.status == "failed"
    assert "TTSUnavailableError" in store.job.error_message


@pytest.mark.asyncio
async def test_invalid_voice_fails_immediately_without_retry(monkeypatch, tmp_path):
    store = FakeStore(attempts=1)
    store.install(monkeypatch)

    outcome = await run(store, tmp_path, tts=FakeTTS(fail_from_call=0, error=TTSInvalidVoiceError("voix inconnue")))

    assert outcome == "failed" and store.requeues == []


@pytest.mark.asyncio
async def test_course_without_usable_content_fails_permanently(monkeypatch, tmp_path):
    empty = {**COURSE, "summary": " ", "next_steps": [],
             "answer": {"quoi": "", "pourquoi": "", "comment": "",
                        "worked_example": {"statement": "", "steps": [], "result": ""}, "key_points": []}}
    store = FakeStore(course=empty)
    store.install(monkeypatch)

    outcome = await run(store, tmp_path)

    assert outcome == "failed" and "exploitable" in store.job.error_message and store.requeues == []


@pytest.mark.asyncio
async def test_missing_course_session_fails(monkeypatch, tmp_path):
    store = FakeStore(course=None)
    store.install(monkeypatch)

    assert await run(store, tmp_path) == "failed"


@pytest.mark.asyncio
async def test_graceful_stop_requeues_without_counting_attempt(monkeypatch, tmp_path):
    store = FakeStore()
    store.install(monkeypatch)

    outcome = await run(store, tmp_path, should_stop=lambda: True)

    assert outcome == "interrupted"
    assert store.requeues == [{"error": None, "refund": True}]
    assert store.job.script is not None                                            # checkpoint conservé


@pytest.mark.asyncio
async def test_unknown_job_raises(monkeypatch, tmp_path):
    store = FakeStore()
    store.install(monkeypatch)

    async def none(db, job_id):
        return None

    monkeypatch.setattr(pipeline.jobs_repo, "get_by_id", none)
    with pytest.raises(pipeline.JobNotFoundError):
        await run(store, tmp_path)


def test_spoken_turns_open_a_chapter_per_block():
    script = PodcastScript.model_validate({
        "title": "T",
        "intro_turns": [{"speaker": "HOST", "text": "a"}, {"speaker": "EXPERT", "text": "b"}],
        "segments": [{"section_ref": 1, "title": "Notion", "turns": [
            {"speaker": "HOST", "text": "c"}, {"speaker": "EXPERT", "text": "d"}]}],
        "outro_turns": [{"speaker": "HOST", "text": "e"}],
    })
    assert [t.chapter for t in build_spoken_turns(script)] == ["Introduction", None, "Notion", None, "Conclusion"]


# ─── Jobs : paramètres, idempotence, chemins ─────────────────


def test_params_hash_is_stable_and_sensitive():
    a = compute_params_hash({"style": "concise", "target_minutes": 5})
    assert a == compute_params_hash({"target_minutes": 5, "style": "concise"})
    assert a != compute_params_hash({"style": "concise", "target_minutes": 6})
    assert len(a) == 64


def test_resolve_params_applies_defaults_and_caps():
    settings = Settings(gemini_api_key="k", podcast_default_target_minutes=10, podcast_max_minutes=20)
    assert resolve_params(None, settings) == {"style": "conversational", "target_minutes": 10}
    assert resolve_params(PodcastGenerationRequest(target_minutes=45), settings)["target_minutes"] == 20


def test_job_dir_stays_inside_storage_and_rejects_traversal(tmp_path):
    settings = settings_for(tmp_path)
    job_id = uuid.uuid4()
    assert job_dir(settings, job_id).parent == (tmp_path / "podcasts").resolve()
    with pytest.raises(ValueError):
        job_dir(settings, "../../etc")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_enqueue_is_idempotent_unless_forced(monkeypatch, tmp_path):
    existing = SimpleNamespace(id=uuid.uuid4())
    created = SimpleNamespace(id=uuid.uuid4())
    monkeypatch.setattr(jobs_module.course_session_repository, "get_by_id", AsyncMock(return_value=object()))
    monkeypatch.setattr(jobs_module.podcast_job_repository, "find_reusable", AsyncMock(return_value=existing))
    create = AsyncMock(return_value=created)
    monkeypatch.setattr(jobs_module.podcast_job_repository, "create", create)
    settings = settings_for(tmp_path)
    sid = uuid.uuid4()

    job, is_new = await enqueue_podcast_job(FakeSessionFactory(), sid, None, settings)
    assert (job, is_new) == (existing, False) and create.await_count == 0

    job, is_new = await enqueue_podcast_job(FakeSessionFactory(), sid, PodcastGenerationRequest(force=True), settings)
    assert (job, is_new) == (created, True) and create.await_count == 1
    assert create.await_args.kwargs["params_hash"] == compute_params_hash(resolve_params(None, settings))


@pytest.mark.asyncio
async def test_enqueue_unknown_session(monkeypatch, tmp_path):
    monkeypatch.setattr(jobs_module.course_session_repository, "get_by_id", AsyncMock(return_value=None))
    with pytest.raises(CourseSessionNotFoundError):
        await enqueue_podcast_job(FakeSessionFactory(), uuid.uuid4(), None, settings_for(tmp_path))


# ─── Worker et CLI ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_worker_processes_one_job_or_reports_empty_queue(monkeypatch, tmp_path):
    settings = settings_for(tmp_path)
    monkeypatch.setattr(podcast_worker.jobs_repo, "release_stale", AsyncMock(return_value=0))
    claim = AsyncMock(side_effect=[SimpleNamespace(id=uuid.uuid4()), None])
    monkeypatch.setattr(podcast_worker.jobs_repo, "claim_next", claim)
    runner = AsyncMock(return_value="done")
    monkeypatch.setattr(podcast_worker, "run_podcast_job", runner)

    args = (FakeSessionFactory(), AsyncMock(), FakeTTS(), settings)
    assert await podcast_worker.process_next_job(*args) is True
    assert await podcast_worker.process_next_job(*args) is False
    assert runner.await_count == 1


def test_purge_removes_only_expired_dirs(tmp_path):
    settings = settings_for(tmp_path, podcast_retention_days=30)
    root = Path(settings.podcast_storage_dir)
    old, recent = root / "old", root / "recent"
    old.mkdir(parents=True)
    recent.mkdir()
    (old / "podcast.mp3").write_bytes(b"x")
    past = time.time() - 40 * 86400
    os.utime(old, (past, past))

    assert podcast_worker.purge_expired_dirs(settings) == 1
    assert not old.exists() and recent.exists()


def test_purge_disabled_when_retention_is_zero(tmp_path):
    settings = settings_for(tmp_path, podcast_retention_days=0)
    Path(settings.podcast_storage_dir).mkdir(parents=True)
    assert podcast_worker.purge_expired_dirs(settings) == 0


def test_cli_parses_podcast_commands():
    parser = build_parser()
    sid = uuid.uuid4()
    args = parser.parse_args(["podcast", "enqueue", str(sid), "--force", "--style", "concise", "--minutes", "8"])
    assert (args.command, args.session_id, args.force, args.style, args.minutes) == ("enqueue", sid, True, "concise", 8)
    assert parser.parse_args(["podcast", "run", str(sid)]).job_id == sid
    with pytest.raises(SystemExit):
        parser.parse_args(["podcast", "status", "pas-un-uuid"])

