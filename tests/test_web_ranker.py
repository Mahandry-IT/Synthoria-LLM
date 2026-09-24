from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.core.exceptions import GeminiUnavailableError
from app.services.media.storage import NormalizedImage
from app.services.media.web_ranker import rank_web_images


def _image(tag: str = "a") -> NormalizedImage:
    return NormalizedImage(content=f"bytes-{tag}".encode(), sha256=f"sha-{tag}", mime="image/webp", width=800, height=600)


def _candidates(n: int) -> list[tuple[NormalizedImage, str]]:
    return [(_image(str(i)), f"Titre {i}") for i in range(n)]


@pytest.mark.asyncio
async def test_returns_none_for_empty_candidates():
    client = AsyncMock()
    result = await rank_web_images(
        [], image_alt="a", image_query="q", section_title="s", gemini_client=client, settings=Settings(),
    )
    assert result is None
    client.rank_images.assert_not_awaited()


@pytest.mark.asyncio
async def test_skips_gemini_call_when_verify_disabled():
    client = AsyncMock()
    settings = Settings(media_web_verify_enabled=False)

    result = await rank_web_images(
        _candidates(3), image_alt="a", image_query="q", section_title="s", gemini_client=client, settings=settings,
    )

    assert result == 0  # premier candidat, déjà filtré en amont
    client.rank_images.assert_not_awaited()


@pytest.mark.asyncio
async def test_returns_best_index_above_threshold():
    client = AsyncMock()
    client.rank_images.return_value = {"best_index": 2, "score": 80, "reason": "correspond bien"}
    settings = Settings(media_web_verify_min_score=40)

    result = await rank_web_images(
        _candidates(3), image_alt="chat noir", image_query="black cat", section_title="s",
        gemini_client=client, settings=settings,
    )

    assert result == 2
    client.rank_images.assert_awaited_once()


@pytest.mark.asyncio
async def test_returns_none_when_best_index_is_null():
    client = AsyncMock()
    client.rank_images.return_value = {"best_index": None, "score": 10, "reason": "hors sujet"}

    result = await rank_web_images(
        _candidates(3), image_alt="a", image_query="q", section_title="s", gemini_client=client, settings=Settings(),
    )

    assert result is None


@pytest.mark.asyncio
async def test_returns_none_when_score_below_threshold():
    client = AsyncMock()
    client.rank_images.return_value = {"best_index": 0, "score": 20, "reason": "faible"}
    settings = Settings(media_web_verify_min_score=40)

    result = await rank_web_images(
        _candidates(3), image_alt="a", image_query="q", section_title="s", gemini_client=client, settings=settings,
    )

    assert result is None


@pytest.mark.asyncio
async def test_rejects_out_of_range_index_as_no_candidate():
    """Gemini ne doit jamais pouvoir pointer hors de la liste fournie — un index halluciné est traité
    comme aucune sélection, jamais comme un candidat arbitraire."""
    client = AsyncMock()
    client.rank_images.return_value = {"best_index": 99, "score": 90, "reason": "?"}

    result = await rank_web_images(
        _candidates(3), image_alt="a", image_query="q", section_title="s", gemini_client=client, settings=Settings(),
    )

    assert result is None


@pytest.mark.asyncio
async def test_falls_back_to_first_candidate_on_gemini_failure():
    client = AsyncMock()
    client.rank_images.side_effect = GeminiUnavailableError("indisponible")

    result = await rank_web_images(
        _candidates(3), image_alt="a", image_query="q", section_title="s", gemini_client=client, settings=Settings(),
    )

    assert result == 0


@pytest.mark.asyncio
async def test_falls_back_to_first_candidate_on_invalid_response():
    client = AsyncMock()
    client.rank_images.return_value = {"score": "not-a-number"}  # schéma invalide

    result = await rank_web_images(
        _candidates(3), image_alt="a", image_query="q", section_title="s", gemini_client=client, settings=Settings(),
    )

    assert result == 0
