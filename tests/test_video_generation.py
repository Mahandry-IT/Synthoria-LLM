"""Tests for video generation services and endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.config import Settings
from app.core.exceptions import (
    HFVideoGenerationError,
    HFVideoUnavailableError,
)
from app.services.video_generator import (
    Scene,
    build_video_prompt_from_course,
)


# --- build_video_prompt_from_course ---


def _make_course_response(sections=None, pitfalls=None, title="Transformateur", subject="Électrotechnique"):
    """Construit un gemini_response minimal pour les tests."""
    if sections is None:
        sections = [
            {
                "title": "Le transformateur",
                "quoi": "Un transformateur est un dispositif qui convertit les niveaux de tension.",
                "pourquoi": "Il permet de transmettre l'énergie sur de longues distances.",
                "comment": "Il utilise deux bobines couplées magnétiquement.",
            }
        ]
    return {
        "meta": {"title": title, "subject": subject, "language": "fr"},
        "sections": sections,
        "common_pitfalls": pitfalls or [],
        "summary": "Résumé du cours.",
    }


def test_build_video_prompt_returns_scenes():
    response = _make_course_response()
    scenes = build_video_prompt_from_course(response)
    assert len(scenes) >= 2  # Au moins accroche + conclusion
    assert all(isinstance(s, Scene) for s in scenes)
    assert all(5 <= s.duration_seconds <= 10 for s in scenes)


def test_build_video_prompt_first_scene_is_hook():
    response = _make_course_response()
    scenes = build_video_prompt_from_course(response)
    assert "Transformateur" in scenes[0].narration


def test_build_video_prompt_last_scene_is_summary():
    response = _make_course_response()
    scenes = build_video_prompt_from_course(response)
    assert "résumé" in scenes[-1].narration.lower() or "résumé" in scenes[-1].visual_prompt.lower()


def test_build_video_prompt_with_pitfalls():
    pitfalls = [{"description": "Erreur courante", "how_to_avoid": "Vérifier les unités"}]
    response = _make_course_response(pitfalls=pitfalls)
    scenes = build_video_prompt_from_course(response)
    pitfall_scene = [s for s in scenes if "piège" in s.narration.lower() or "erreur" in s.narration.lower()]
    assert len(pitfall_scene) == 1


def test_build_video_prompt_empty_sections():
    response = _make_course_response(sections=[])
    scenes = build_video_prompt_from_course(response)
    # Au moins accroche + conclusion
    assert len(scenes) >= 2


def test_build_video_prompt_truncates_long_content():
    long_text = "Ceci est un texte très long. " * 200
    sections = [
        {
            "title": "Section longue",
            "quoi": long_text,
            "pourquoi": long_text,
            "comment": long_text,
        }
    ]
    response = _make_course_response(sections=sections)
    scenes = build_video_prompt_from_course(response)
    # Le narration ne doit pas dépasser ~250 chars
    for s in scenes:
        assert len(s.narration) < 500


def test_build_video_prompt_subsection_fallback():
    """Format Gemini brut (subsections) au lieu de quoi/pourquoi/comment."""
    sections = [
        {
            "title": "Les matrices",
            "subsections": [
                {"title": "Quoi", "blocks": [{"type": "text", "text": "Définition des matrices"}]},
                {"title": "Pourquoi", "blocks": [{"type": "text", "text": "Elles sont utiles"}]},
                {"title": "Comment", "blocks": [{"type": "text", "text": "Multiplication matricielle"}]},
            ],
        }
    ]
    response = _make_course_response(sections=sections)
    scenes = build_video_prompt_from_course(response)
    assert any("matrice" in s.narration.lower() for s in scenes)


# --- HFVideoClient (unit tests with mocks) ---


@pytest.mark.asyncio
async def test_hf_video_client_requires_api_token():
    from app.services.hf_video_client import HFVideoClient
    settings = Settings(hf_api_token=None)
    client = HFVideoClient(settings)
    with pytest.raises(HFVideoUnavailableError, match="HF_API_TOKEN"):
        await client.generate_video("test prompt")


@pytest.mark.asyncio
async def test_hf_video_client_generate_video_calls_api():
    from app.services.hf_video_client import HFVideoClient
    settings = Settings(hf_api_token="fake-token", hf_video_model_primary="test-model")
    client = HFVideoClient(settings)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "video/mp4"}
    mock_response.content = b"fake-video-bytes"

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
        video = await client.generate_video("A whiteboard with math")
        assert video == b"fake-video-bytes"
        mock_post.assert_awaited_once()


@pytest.mark.asyncio
async def test_hf_video_client_retry_on_503_then_success():
    from app.services.hf_video_client import HFVideoClient, _POLL_INTERVAL_SECONDS
    settings = Settings(hf_api_token="fake-token", hf_video_max_retries=2, hf_video_model_primary="test-model")
    client = HFVideoClient(settings)

    loading_response = MagicMock()
    loading_response.status_code = 503
    loading_response.json.return_value = {"estimated_time": 0.1}

    success_response = MagicMock()
    success_response.status_code = 200
    success_response.headers = {"content-type": "video/mp4"}
    success_response.content = b"video-bytes"

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=[loading_response, success_response]) as mock_post:
        with patch("asyncio.sleep", new_callable=AsyncMock):
            video = await client.generate_video("prompt")
            assert video == b"video-bytes"
            assert mock_post.await_count == 2


@pytest.mark.asyncio
async def test_hf_video_client_fallback():
    from app.services.hf_video_client import HFVideoClient
    settings = Settings(
        hf_api_token="fake-token",
        hf_video_model_primary="primary-model",
        hf_video_model_fallback="fallback-model",
    )
    client = HFVideoClient(settings)

    with patch.object(client, "generate_video", new_callable=AsyncMock) as mock_gen:
        mock_gen.side_effect = [
            HFVideoGenerationError("Primary failed"),
            (b"fallback-video", "fallback-model", False),
        ]
        # Override to return the right format
        async def _gen_side_effect(prompt, model=None):
            if model == "primary-model":
                raise HFVideoGenerationError("Primary failed")
            return b"fallback-video"

        mock_gen.side_effect = _gen_side_effect
        video, model, fallback = await client.generate_video_with_fallback("prompt")
        assert video == b"fallback-video"
        assert model == "fallback-model"
        assert fallback is True


# --- Endpoint tests (integration-like) ---


class _AsyncSessionContextManager:
    """Mock d'un async context manager pour async_sessionmaker()."""

    def __init__(self, db_session):
        self._db = db_session

    async def __aenter__(self):
        return self._db

    async def __aexit__(self, *args):
        return False


def _mock_request_app(db_session=None):
    """Construit un mock de request.app.state avec session_factory."""
    app_state = MagicMock()
    mock_db = db_session or AsyncMock()
    session_factory = MagicMock(return_value=_AsyncSessionContextManager(mock_db))
    app_state.db_session_factory = session_factory
    request = MagicMock()
    request.app.state = app_state
    return request, session_factory, mock_db


@pytest.mark.asyncio
async def test_video_generate_endpoint_404_session_not_found():
    from app.api.routes import generate_video
    from uuid import uuid4
    from fastapi import HTTPException

    request, session_factory, mock_db = _mock_request_app()

    # Course session not found
    with patch("app.repositories.course_session_repository.get_by_id", new_callable=AsyncMock, return_value=None):
        with pytest.raises(HTTPException) as exc_info:
            await generate_video(uuid4(), request, MagicMock())
        assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_video_generate_endpoint_409_conflict():
    from app.api.routes import generate_video
    from uuid import uuid4
    from fastapi import HTTPException
    from app.db.models import CourseSession, VideoGenerationJob

    request, session_factory, mock_db = _mock_request_app()

    fake_session_id = uuid4()
    fake_course_session = MagicMock(spec=CourseSession)
    fake_course_session.id = fake_session_id

    existing_job = MagicMock(spec=VideoGenerationJob)
    existing_job.id = uuid4()

    with patch("app.repositories.course_session_repository.get_by_id", new_callable=AsyncMock, return_value=fake_course_session):
        with patch("app.repositories.video_job_repository.get_active_job_for_session", new_callable=AsyncMock, return_value=existing_job):
            with pytest.raises(HTTPException) as exc_info:
                await generate_video(fake_session_id, request, MagicMock())
            assert exc_info.value.status_code == 409
