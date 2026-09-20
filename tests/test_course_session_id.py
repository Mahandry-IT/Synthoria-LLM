import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.api.routes import _persist_course_session
from app.api.schemas import CourseGenerationResponse
from app.repositories import course_session_repository

RESPONSE = CourseGenerationResponse.model_validate({
    "mode": "file_question",
    "format": "focused_answer",
    "meta": {"title": "T", "subject": "S", "language": "fr", "generated_at": "2026-08-18T10:00:00Z"},
    "sources": [],
    "answer": {
        "quoi": "q", "pourquoi": "p", "comment": "c",
        "worked_example": {"statement": "s", "steps": [], "result": "r"},
        "key_points": [],
    },
    "summary": "résumé",
    "next_steps": [],
    "session_id": str(uuid.uuid4()),
    "podcast_job_id": str(uuid.uuid4()),
})


@pytest.mark.asyncio
async def test_save_does_not_persist_session_and_podcast_ids():
    session = AsyncMock()
    session.add = MagicMock()

    row = await course_session_repository.save(
        session, question="Q", filenames=[], mode="file_question", response=RESPONSE
    )

    assert "session_id" not in row.gemini_response
    assert "podcast_job_id" not in row.gemini_response
    assert row.gemini_response["summary"] == "résumé"


def _request() -> SimpleNamespace:
    factory = MagicMock()
    factory.return_value.__aenter__ = AsyncMock(return_value=AsyncMock())
    factory.return_value.__aexit__ = AsyncMock(return_value=False)
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(db_session_factory=factory)))


@pytest.mark.asyncio
async def test_persist_returns_session_id():
    sid = uuid.uuid4()
    with patch("app.api.routes.course_session_repository.save", new_callable=AsyncMock) as save:
        save.return_value = SimpleNamespace(id=sid)
        result = await _persist_course_session(
            _request(), question="Q", filenames=[], mode="file_question", response=RESPONSE
        )
    assert result == sid


@pytest.mark.asyncio
async def test_persist_failure_returns_none_and_never_raises():
    with patch("app.api.routes.course_session_repository.save", new_callable=AsyncMock) as save:
        save.side_effect = RuntimeError("db down")
        result = await _persist_course_session(
            _request(), question="Q", filenames=[], mode="file_question", response=RESPONSE
        )
    assert result is None
