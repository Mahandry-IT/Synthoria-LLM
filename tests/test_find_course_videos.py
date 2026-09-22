from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.core.exceptions import YoutubeQuotaExceeded, YoutubeUnavailable
from app.services import course_videos
from app.services.course_videos import find_course_videos, reset_quota_breaker
from app.services.youtube_data_client import YoutubeVideoDetails


@pytest.fixture(autouse=True)
def _clean_breaker():
    """Le disjoncteur de quota est un singleton de module : jamais de fuite entre les tests."""
    reset_quota_breaker()
    yield
    reset_quota_breaker()


def _settings(**overrides) -> Settings:
    return Settings(
        youtube_api_key="fake-key",
        youtube_max_queries=2,
        course_videos_max=3,
        youtube_min_duration_seconds=60,
        youtube_max_duration_seconds=3600,
        **overrides,
    )


def _details(video_id: str, *, embeddable: bool = True) -> YoutubeVideoDetails:
    return YoutubeVideoDetails(
        video_id=video_id, title=f"Titre {video_id}", channel_title="Chaîne", description="",
        published_at="2024-01-01T00:00:00Z", duration_iso8601="PT10M",
        embeddable=embeddable, privacy_status="public", live_broadcast_content="none",
    )


class FakeDataClient:
    """Simule YoutubeDataClient : `search_by_query` associe une requête à ses IDs (dans l'ordre)."""

    def __init__(self, search_by_query: dict[str, list[str]], details_by_id: dict[str, YoutubeVideoDetails]):
        self.search_by_query = search_by_query
        self.details_by_id = details_by_id
        self.search_calls: list[str] = []
        self.details_calls: list[list[str]] = []

    async def search(self, query: str, *, max_results: int = 8) -> list[str]:
        self.search_calls.append(query)
        return self.search_by_query.get(query, [])

    async def details(self, video_ids: list[str]) -> list[YoutubeVideoDetails]:
        self.details_calls.append(video_ids)
        return [self.details_by_id[v] for v in video_ids if v in self.details_by_id]


def _session_factory():
    @asynccontextmanager
    async def factory():
        yield "db-handle"

    return factory


@pytest.mark.asyncio
async def test_no_api_key_returns_empty_without_creating_a_client():
    result = await find_course_videos(["q"], "Sujet", Settings(youtube_api_key=None))

    assert result == []


@pytest.mark.asyncio
async def test_open_breaker_skips_the_call_entirely():
    course_videos._breaker.trip()  # noqa: SLF001 — précondition du test
    client = FakeDataClient({}, {})

    result = await find_course_videos(["q"], "Sujet", _settings(), client=client)

    assert result == [] and client.search_calls == []


@pytest.mark.asyncio
async def test_quota_exceeded_opens_the_breaker_and_returns_empty():
    class QuotaClient(FakeDataClient):
        async def search(self, query: str, *, max_results: int = 8) -> list[str]:
            raise YoutubeQuotaExceeded("nope")

    result = await find_course_videos(["q"], "Sujet", _settings(), client=QuotaClient({}, {}))

    assert result == []
    assert course_videos._breaker.is_open() is True  # noqa: SLF001


@pytest.mark.asyncio
async def test_other_service_error_returns_empty_without_opening_breaker():
    class DownClient(FakeDataClient):
        async def search(self, query: str, *, max_results: int = 8) -> list[str]:
            raise YoutubeUnavailable("down")

    result = await find_course_videos(["q"], "Sujet", _settings(), client=DownClient({}, {}))

    assert result == []
    assert course_videos._breaker.is_open() is False  # noqa: SLF001


@pytest.mark.asyncio
async def test_empty_queries_use_fallback_queries(monkeypatch):
    client = FakeDataClient(
        {"Sujet cours": ["v11111111vv"]},
        {"v11111111vv": _details("v11111111vv")},
    )

    result = await find_course_videos([], "Sujet", _settings(), client=client)

    assert "Sujet cours" in client.search_calls
    assert [v.video_id for v in result] == ["v11111111vv"]


@pytest.mark.asyncio
async def test_cache_hit_makes_no_network_call(monkeypatch):
    get_fresh = AsyncMock(return_value=[{"video_id": "cached11111", "title": "Cache", "channel_title": "C",
                                          "duration_seconds": 300, "published_at": "2024-01-01T00:00:00Z"}])
    monkeypatch.setattr(course_videos.cache_repo, "get_fresh", get_fresh)
    client = FakeDataClient({}, {})

    result = await find_course_videos(["cours de thermo"], "Sujet", _settings(), _session_factory(), client=client)

    assert [v.video_id for v in result] == ["cached11111"]
    assert client.search_calls == [] and client.details_calls == []


@pytest.mark.asyncio
async def test_cache_miss_fetches_then_writes_the_cache(monkeypatch):
    monkeypatch.setattr(course_videos.cache_repo, "get_fresh", AsyncMock(return_value=None))
    upsert = AsyncMock()
    monkeypatch.setattr(course_videos.cache_repo, "upsert", upsert)
    client = FakeDataClient({"cours de thermo": ["v11111111vv"]}, {"v11111111vv": _details("v11111111vv")})

    result = await find_course_videos(["cours de thermo"], "Sujet", _settings(), _session_factory(), client=client)

    assert [v.video_id for v in result] == ["v11111111vv"]
    upsert.assert_awaited_once()
    cached_candidates = upsert.await_args.args[-1]
    assert cached_candidates == [{
        "video_id": "v11111111vv", "title": "Titre v11111111vv", "channel_title": "Chaîne",
        "duration_seconds": 600, "published_at": "2024-01-01T00:00:00Z",
    }]


@pytest.mark.asyncio
async def test_without_session_factory_calls_api_without_touching_the_cache(monkeypatch):
    get_fresh = AsyncMock()
    upsert = AsyncMock()
    monkeypatch.setattr(course_videos.cache_repo, "get_fresh", get_fresh)
    monkeypatch.setattr(course_videos.cache_repo, "upsert", upsert)
    client = FakeDataClient({"q": ["v11111111vv"]}, {"v11111111vv": _details("v11111111vv")})

    result = await find_course_videos(["q"], "Sujet", _settings(), db_session_factory=None, client=client)

    assert [v.video_id for v in result] == ["v11111111vv"]
    get_fresh.assert_not_awaited()
    upsert.assert_not_awaited()


@pytest.mark.asyncio
async def test_filters_drop_non_embeddable_candidates(monkeypatch):
    client = FakeDataClient(
        {"q": ["good1111111", "bad11111111"]},
        {"good1111111": _details("good1111111"), "bad11111111": _details("bad11111111", embeddable=False)},
    )

    result = await find_course_videos(["q"], "Sujet", _settings(), client=client)

    assert [v.video_id for v in result] == ["good1111111"]


@pytest.mark.asyncio
async def test_two_queries_round_robin_and_a_single_grouped_details_call():
    client = FakeDataClient(
        {"q1": ["a1111111111", "a2222222222"], "q2": ["b1111111111"]},
        {
            "a1111111111": _details("a1111111111"),
            "a2222222222": _details("a2222222222"),
            "b1111111111": _details("b1111111111"),
        },
    )

    result = await find_course_videos(["q1", "q2"], "Sujet", Settings(youtube_api_key="fake-key", course_videos_max=5, youtube_min_duration_seconds=60, youtube_max_duration_seconds=3600), client=client)

    assert [v.video_id for v in result] == ["a1111111111", "b1111111111", "a2222222222"]
    assert len(client.details_calls) == 1  # un seul appel details() groupé pour les 2 requêtes


@pytest.mark.asyncio
async def test_result_is_capped_at_course_videos_max():
    client = FakeDataClient(
        {"q": ["a1111111111", "a2222222222", "a3333333333"]},
        {v: _details(v) for v in ["a1111111111", "a2222222222", "a3333333333"]},
    )

    result = await find_course_videos(["q"], "Sujet", Settings(youtube_api_key="fake-key", course_videos_max=2, youtube_min_duration_seconds=60, youtube_max_duration_seconds=3600), client=client)

    assert len(result) == 2
