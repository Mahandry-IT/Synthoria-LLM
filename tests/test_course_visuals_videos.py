from datetime import datetime, timezone
from unittest.mock import AsyncMock

import httpx
import pytest

from app.core.config import Settings
from app.schemas.course_generation import CourseGenerationSchema
from app.services import youtube
from app.services.course_generator import _map_schema_to_response
from app.services.course_videos import attach_verified_videos

VID = "dQw4w9WgXcQ"


@pytest.mark.parametrize("url", [
    f"https://www.youtube.com/watch?v={VID}&t=3",
    f"https://youtu.be/{VID}",
    f"https://www.youtube.com/embed/{VID}",
    f"https://m.youtube.com/shorts/{VID}",
])
def test_extract_video_id_accepts_youtube_urls(url):
    assert youtube.extract_video_id(url) == VID


@pytest.mark.parametrize("url", ["https://example.com/watch?v=" + VID, "https://www.youtube.com/watch?v=short", "nope"])
def test_extract_video_id_rejects_others(url):
    assert youtube.extract_video_id(url) is None


@pytest.mark.asyncio
async def test_verify_one_keeps_existing_and_drops_missing():
    def handler(request: httpx.Request) -> httpx.Response:
        if VID in request.url.params["url"]:
            return httpx.Response(200, json={"title": "Vrai titre", "author_name": "Chaîne"})
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        ok = await youtube._verify_one(client, youtube.candidate_video(f"https://youtu.be/{VID}"))
        ko = await youtube._verify_one(client, youtube.candidate_video("https://youtu.be/AAAAAAAAAAA"))

    assert ok is not None and ok.title == "Vrai titre" and ok.channel == "Chaîne"
    assert ko is None


def _schema(sections, queries=()):
    return CourseGenerationSchema.model_validate({
        "mode": "question_only", "format": "focused_answer",
        "meta": {"title": "T", "subject": "S", "generated_at": datetime.now(timezone.utc).isoformat()},
        "sections": sections, "confidence": "high",
        "video_search_queries": list(queries),
    })


def test_tables_exposed_structured_and_not_flattened():
    schema = _schema([{
        "type": "development", "title": "A",
        "subsections": [
            {"title": "Quoi", "blocks": [
                {"type": "text", "text": "Définition."},
                {"type": "table", "table": {"caption": "Comparatif", "headers": ["a", "b"], "rows": [["1", "2"]]}},
            ]},
            {"title": "Pourquoi", "blocks": [{"type": "text", "text": "Car."}]},
            {"title": "Comment", "blocks": [{"type": "text", "text": "Ainsi."}]},
        ],
    }])

    response = _map_schema_to_response(schema)

    section = response.sections[0]
    assert section.quoi == "Définition."
    assert section.tables[0].caption == "Comparatif"
    assert section.tables[0].rows == [["1", "2"]]
    assert response.videos == []  # jamais depuis Gemini : attaché après coup par attach_verified_videos


@pytest.mark.asyncio
async def test_attach_videos_falls_back_to_web_search_without_a_youtube_api_key(monkeypatch):
    """Sans clé YOUTUBE_API_KEY, find_course_videos() est un no-op : repli grounding + oEmbed."""
    settings = Settings(gemini_api_key="k", course_videos_enabled=True, youtube_api_key=None)
    response = _map_schema_to_response(_schema([{
        "type": "development", "title": "A", "blocks": [{"type": "text", "text": "x"}],
    }]))
    gemini = AsyncMock()
    gemini.search_grounded.return_value = (f"Voir https://www.youtube.com/watch?v={VID}", [])

    async def fake_verify(candidates, **_):
        return [v for v in candidates if v.video_id == VID]

    monkeypatch.setattr("app.services.course_videos.verify_videos", fake_verify)

    result = await attach_verified_videos(response, settings, gemini)

    assert [v.video_id for v in result.videos] == [VID]
    assert result.videos[0].section_id is None  # repli sans classement : vidéo générale
    gemini.search_grounded.assert_awaited_once()


@pytest.mark.asyncio
async def test_attach_videos_links_ranked_videos_to_their_section(monkeypatch):
    settings = Settings(gemini_api_key="k", course_videos_enabled=True, youtube_api_key="yt")
    response = _map_schema_to_response(_schema([
        {"type": "development", "title": "A", "blocks": [{"type": "text", "text": "x"}]},
        {"type": "development", "title": "B", "blocks": [{"type": "text", "text": "y"}]},
    ]))
    other = "BBBBBBBBBBB"

    async def fake_find(*_args, **_kwargs):
        return [youtube.candidate_video(f"https://youtu.be/{vid}") for vid in (VID, other)]

    monkeypatch.setattr("app.services.course_videos.find_course_videos", fake_find)
    gemini = AsyncMock()
    gemini.is_degraded.return_value = False
    gemini.format_structured.return_value = {"items": [
        {"candidate_index": 0, "category": "cours", "level": "debutant", "relevance_score": 90,
         "reason": "Clair.", "section_index": 1},
        {"candidate_index": 1, "category": "methode", "level": "debutant", "relevance_score": 80, "reason": "Utile."},
    ]}

    result = await attach_verified_videos(response, settings, gemini)

    assert {v.video_id: v.section_id for v in result.videos} == {VID: response.sections[1].id, other: None}


@pytest.mark.asyncio
async def test_attach_videos_disabled_returns_empty():
    response = _map_schema_to_response(_schema([{
        "type": "development", "title": "A", "blocks": [{"type": "text", "text": "x"}],
    }]))
    result = await attach_verified_videos(response, Settings(gemini_api_key="k", course_videos_enabled=False), AsyncMock())
    assert result.videos == []
