from unittest.mock import AsyncMock

import httpx
import pytest

from app.core.config import Settings
from app.services import youtube
from app.services.course_generator import _search_videos, attach_verified_videos, _map_schema_to_response
from app.services.youtube import resolve_grounding_video_ids
from tests.test_course_visuals_videos import _schema

VID = "dQw4w9WgXcQ"
REDIRECT = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/abc"


def _client_returning(location_by_url: dict[str, str | Exception]):
    """Faux httpx.AsyncClient : renvoie un en-tête Location (ou lève) selon l'URL demandée."""

    class FakeClient:
        def __init__(self, *_, **__):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def get(self, url, **kwargs):
            assert kwargs.get("follow_redirects") is False  # un seul saut, jamais de suivi libre
            outcome = location_by_url[url]
            if isinstance(outcome, Exception):
                raise outcome
            return httpx.Response(302, headers={"location": outcome})

    return FakeClient


@pytest.mark.asyncio
async def test_resolves_direct_youtube_sources_without_any_request(monkeypatch):
    monkeypatch.setattr(youtube.httpx, "AsyncClient", _client_returning({}))

    ids = await resolve_grounding_video_ids([{"reference": f"https://www.youtube.com/watch?v={VID}"}])

    assert ids == [VID]


@pytest.mark.asyncio
async def test_resolves_grounding_redirect_to_video_id(monkeypatch):
    monkeypatch.setattr(
        youtube.httpx, "AsyncClient", _client_returning({REDIRECT: f"https://www.youtube.com/watch?v={VID}"})
    )

    assert await resolve_grounding_video_ids([{"reference": REDIRECT}, {"reference": REDIRECT}]) == [VID]


@pytest.mark.asyncio
async def test_never_follows_arbitrary_hosts_and_ignores_non_video_targets(monkeypatch):
    monkeypatch.setattr(
        youtube.httpx, "AsyncClient", _client_returning({REDIRECT: "https://example.com/article"})
    )
    sources = [{"reference": "http://169.254.169.254/latest/meta-data"}, {"reference": "https://example.com/x"},
               {"reference": REDIRECT}]

    assert await resolve_grounding_video_ids(sources) == []


@pytest.mark.asyncio
async def test_unreachable_redirect_is_skipped_not_raised(monkeypatch):
    monkeypatch.setattr(youtube.httpx, "AsyncClient", _client_returning({REDIRECT: httpx.ConnectError("boom")}))

    assert await resolve_grounding_video_ids([{"reference": REDIRECT}]) == []


@pytest.mark.asyncio
async def test_search_videos_uses_citation_sources_when_text_has_no_url(monkeypatch):
    monkeypatch.setattr(
        youtube.httpx, "AsyncClient", _client_returning({REDIRECT: f"https://youtu.be/{VID}"})
    )
    gemini = AsyncMock()
    gemini.search_grounded.return_value = ("Voici de bonnes vidéos, sans lien dans le texte.", [{"reference": REDIRECT}])

    videos = await _search_videos("Transformateur", gemini)

    assert [v.video_id for v in videos] == [VID]


@pytest.mark.asyncio
async def test_invented_ids_are_rejected_then_citations_provide_a_verified_video(monkeypatch):
    settings = Settings(gemini_api_key="k", course_videos_enabled=True)
    response = _map_schema_to_response(_schema([{
        "type": "development", "title": "A", "blocks": [{"type": "text", "text": "x"}],
    }], videos=["https://youtu.be/525c0kWjX7k"]))  # identifiant inventé, 404 chez YouTube
    monkeypatch.setattr(
        youtube.httpx, "AsyncClient", _client_returning({REDIRECT: f"https://www.youtube.com/watch?v={VID}"})
    )
    gemini = AsyncMock()
    gemini.search_grounded.return_value = ("", [{"reference": REDIRECT}])

    async def fake_verify(candidates, **_):
        return [v for v in candidates if v.video_id == VID]

    monkeypatch.setattr("app.services.course_generator.verify_videos", fake_verify)

    result = await attach_verified_videos(response, settings, gemini)

    assert [v.video_id for v in result.videos] == [VID]
