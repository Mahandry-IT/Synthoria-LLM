"""Cible 5 à 10 vidéos (plutôt que 3 fixe) : bornes par défaut, prompt du repli, vivier de classement."""

from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.services.course_videos import _RANKING_POOL_EXTRA, _search_videos, find_course_videos


def test_default_video_count_targets_five_to_ten():
    settings = Settings()

    assert settings.course_videos_min == 5
    assert settings.course_videos_max == 10


@pytest.mark.asyncio
async def test_grounding_fallback_prompt_asks_for_the_configured_range_not_a_fixed_three():
    gemini = AsyncMock()
    gemini.search_grounded.return_value = ("", [])
    settings = Settings(course_videos_min=5, course_videos_max=10)

    await _search_videos("Sujet", gemini, settings)

    prompt = gemini.search_grounded.await_args.kwargs["prompt"]
    assert "entre 5 et 10 vidéos" in prompt
    assert "3 vidéos" not in prompt


@pytest.mark.asyncio
async def test_grounding_fallback_prompt_follows_a_custom_range():
    gemini = AsyncMock()
    gemini.search_grounded.return_value = ("", [])
    settings = Settings(course_videos_min=2, course_videos_max=4)

    await _search_videos("Sujet", gemini, settings)

    assert "entre 2 et 4 vidéos" in gemini.search_grounded.await_args.kwargs["prompt"]


@pytest.mark.asyncio
async def test_data_api_ranking_pool_grows_with_course_videos_max():
    """Le vivier de classement (V2) n'est plus une constante figée : il suit course_videos_max."""

    class FakeDataClient:
        def __init__(self):
            self.max_results_seen: list[int] = []

        async def search(self, query, *, max_results=8):
            self.max_results_seen.append(max_results)
            return [f"a{i:010d}" for i in range(max_results)]

        async def details(self, ids):
            from app.services.youtube_data_client import YoutubeVideoDetails

            return [
                YoutubeVideoDetails(
                    video_id=i, title="T", channel_title="C", description="",
                    published_at="2024-01-01T00:00:00Z", duration_iso8601="PT10M",
                    embeddable=True, privacy_status="public", live_broadcast_content="none",
                )
                for i in ids
            ]

    client = FakeDataClient()
    settings = Settings(
        youtube_api_key="k", course_videos_max=10, course_videos_ranking_enabled=True,
        youtube_search_max_results=20,  # assez de candidats bruts pour remplir le vivier élargi
        youtube_min_duration_seconds=60, youtube_max_duration_seconds=3600,
    )

    result = await find_course_videos(["q"], "Sujet", settings, client=client)

    assert len(result) == 10 + _RANKING_POOL_EXTRA
    assert client.max_results_seen == [20]
