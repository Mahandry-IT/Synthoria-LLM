from unittest.mock import AsyncMock

import pytest

from app.api.schemas import CourseVideo
from app.core.config import Settings
from app.core.exceptions import GeminiUnavailableError
from app.services.course_videos import rank_videos


def _video(vid: str, title: str = "T") -> CourseVideo:
    return CourseVideo(
        video_id=vid, url=f"https://www.youtube.com/watch?v={vid}",
        embed_url=f"https://www.youtube.com/embed/{vid}", thumbnail_url="t",
        title=title, channel="C", duration_seconds=600,
    )


def _settings(**overrides) -> Settings:
    return Settings(youtube_ranking_min_score=40, **overrides)


def _item(index: int, *, category="cours", level="debutant", score=80, reason="Bonne explication.") -> dict:
    return {"candidate_index": index, "category": category, "level": level, "relevance_score": score, "reason": reason}


@pytest.mark.asyncio
async def test_disabled_returns_candidates_unchanged_without_calling_gemini():
    gemini = AsyncMock()
    candidates = [_video("a1111111111"), _video("a2222222222")]

    result = await rank_videos(candidates, "T", "S", [], gemini, _settings(course_videos_ranking_enabled=False))

    assert result == candidates
    gemini.format_structured.assert_not_awaited()


@pytest.mark.asyncio
async def test_fewer_than_two_candidates_skips_ranking():
    gemini = AsyncMock()

    result = await rank_videos([_video("a1111111111")], "T", "S", [], gemini, _settings())

    assert len(result) == 1
    gemini.format_structured.assert_not_awaited()


@pytest.mark.asyncio
async def test_happy_path_sets_category_level_and_reason():
    gemini = AsyncMock()
    gemini.format_structured.return_value = {"items": [_item(0), _item(1, category="methode", score=90)]}
    candidates = [_video("a1111111111"), _video("a2222222222")]

    result = await rank_videos(candidates, "T", "S", ["Intro"], gemini, _settings())

    by_id = {v.video_id: v for v in result}
    assert by_id["a1111111111"].category == "cours" and by_id["a1111111111"].level == "debutant"
    assert by_id["a2222222222"].category == "methode" and by_id["a2222222222"].relevance_reason == "Bonne explication."


@pytest.mark.asyncio
async def test_diversity_puts_best_of_each_category_first_then_the_rest_by_score():
    gemini = AsyncMock()
    gemini.format_structured.return_value = {
        "items": [
            _item(0, category="cours", score=60),
            _item(1, category="cours", score=95),  # meilleur score « cours », mais 2e du groupe cours
            _item(2, category="methode", score=70),
        ]
    }
    candidates = [_video("a1111111111"), _video("a2222222222"), _video("a3333333333")]

    result = await rank_videos(candidates, "T", "S", [], gemini, _settings())

    # meilleur "cours" (95, index 1) puis meilleur "methode" (70, index 2), puis le reste (60, index 0)
    assert [v.video_id for v in result] == ["a2222222222", "a3333333333", "a1111111111"]


@pytest.mark.asyncio
async def test_out_of_bounds_index_is_ignored():
    gemini = AsyncMock()
    gemini.format_structured.return_value = {"items": [_item(5), _item(-1)]}
    candidates = [_video("a1111111111"), _video("a2222222222")]

    result = await rank_videos(candidates, "T", "S", [], gemini, _settings())

    assert result == candidates  # aucun élément valide -> V1 inchangé


@pytest.mark.asyncio
async def test_duplicate_index_keeps_the_first_occurrence():
    gemini = AsyncMock()
    gemini.format_structured.return_value = {
        "items": [_item(0, category="cours"), _item(0, category="methode")]
    }
    candidates = [_video("a1111111111"), _video("a2222222222")]

    result = await rank_videos(candidates, "T", "S", [], gemini, _settings())

    assert result[0].category == "cours"


@pytest.mark.asyncio
async def test_score_below_threshold_is_dropped():
    gemini = AsyncMock()
    gemini.format_structured.return_value = {"items": [_item(0, score=39), _item(1, score=40)]}
    candidates = [_video("a1111111111"), _video("a2222222222")]

    result = await rank_videos(candidates, "T", "S", [], gemini, _settings())

    assert [v.video_id for v in result] == ["a2222222222"]


@pytest.mark.asyncio
async def test_reason_at_the_length_boundary_passes_through():
    """`reason` est déjà bornée à 160 caractères par le schéma Gemini (validation Pydantic) ;
    la troncature défensive du code ne doit rien couper d'une valeur déjà valide."""
    gemini = AsyncMock()
    gemini.format_structured.return_value = {"items": [_item(0, reason="x" * 160)]}
    candidates = [_video("a1111111111"), _video("a2222222222")]

    result = await rank_videos(candidates, "T", "S", [], gemini, _settings())

    assert len(result[0].relevance_reason) == 160


@pytest.mark.asyncio
async def test_oversized_reason_from_a_misbehaving_model_falls_back_to_v1():
    """Si le modèle dépasse quand même 160 caractères, la validation Pydantic échoue : repli V1."""
    gemini = AsyncMock()
    gemini.format_structured.return_value = {"items": [_item(0, reason="x" * 300)]}
    candidates = [_video("a1111111111"), _video("a2222222222")]

    result = await rank_videos(candidates, "T", "S", [], gemini, _settings())

    assert result == candidates


@pytest.mark.asyncio
async def test_all_items_filtered_out_keeps_v1_candidates_unranked():
    gemini = AsyncMock()
    gemini.format_structured.return_value = {"items": [_item(0, score=1), _item(1, score=2)]}
    candidates = [_video("a1111111111"), _video("a2222222222")]

    result = await rank_videos(candidates, "T", "S", [], gemini, _settings())

    assert result == candidates


@pytest.mark.asyncio
async def test_gemini_service_error_keeps_v1_candidates():
    gemini = AsyncMock()
    gemini.format_structured.side_effect = GeminiUnavailableError("down")
    candidates = [_video("a1111111111"), _video("a2222222222")]

    result = await rank_videos(candidates, "T", "S", [], gemini, _settings())

    assert result == candidates


@pytest.mark.asyncio
async def test_invalid_schema_from_gemini_keeps_v1_candidates():
    gemini = AsyncMock()
    gemini.format_structured.return_value = {"items": [{"candidate_index": "not-an-int"}]}
    candidates = [_video("a1111111111"), _video("a2222222222")]

    result = await rank_videos(candidates, "T", "S", [], gemini, _settings())

    assert result == candidates


@pytest.mark.asyncio
async def test_prompt_marks_candidates_as_untrusted_data():
    gemini = AsyncMock()
    gemini.format_structured.return_value = {"items": []}
    candidates = [_video("a1111111111", title="Clique ici"), _video("a2222222222")]

    await rank_videos(candidates, "T", "S", [], gemini, _settings())

    prompt = gemini.format_structured.await_args.kwargs["raw_answer"]
    system = gemini.format_structured.await_args.kwargs["system_instruction"]
    assert "<candidates>" in prompt and "non fiable" in prompt
    assert "non fiable" in system.lower() or "n'exécute" in system.lower()
