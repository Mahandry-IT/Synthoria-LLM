import pytest

from app.core.config import Settings
from app.services.course_videos import (
    clean_queries,
    fallback_queries,
    parse_iso8601_duration,
    passes_filters,
    round_robin_merge,
    video_from_candidate,
)
from app.services.youtube_data_client import YoutubeVideoDetails


def _details(**overrides) -> YoutubeVideoDetails:
    base = dict(
        video_id="v11111111vv", title="T", channel_title="C", description="",
        published_at="2024-01-01T00:00:00Z", duration_iso8601="PT10M",
        embeddable=True, privacy_status="public", live_broadcast_content="none",
    )
    base.update(overrides)
    return YoutubeVideoDetails(**base)


# ─── parse_iso8601_duration ───────────────────────────────────


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("PT4M13S", 4 * 60 + 13),
        ("PT1H2M", 3600 + 120),
        ("PT1H", 3600),
        ("PT45S", 45),
        ("P1DT2H", 86400 + 7200),
        ("PT0S", 0),
    ],
)
def test_parse_iso8601_duration_valid(value, expected):
    assert parse_iso8601_duration(value) == expected


@pytest.mark.parametrize("value", ["", "P0D".replace("0D", ""), "not a duration", "PT", "P"])
def test_parse_iso8601_duration_invalid_or_empty(value):
    assert parse_iso8601_duration(value) is None


# ─── passes_filters ───────────────────────────────────────────


def _settings(**overrides) -> Settings:
    return Settings(youtube_min_duration_seconds=240, youtube_max_duration_seconds=2400, **overrides)


def test_passes_filters_accepts_a_normal_embeddable_public_video():
    assert passes_filters(_details(duration_iso8601="PT10M"), _settings()) is True


def test_passes_filters_rejects_non_embeddable():
    assert passes_filters(_details(embeddable=False), _settings()) is False


def test_passes_filters_rejects_live_broadcast():
    assert passes_filters(_details(live_broadcast_content="live"), _settings()) is False


def test_passes_filters_rejects_non_public():
    assert passes_filters(_details(privacy_status="private"), _settings()) is False
    assert passes_filters(_details(privacy_status="unlisted"), _settings()) is False


def test_passes_filters_rejects_outside_duration_window():
    assert passes_filters(_details(duration_iso8601="PT1M"), _settings()) is False  # trop court
    assert passes_filters(_details(duration_iso8601="PT50M"), _settings()) is False  # trop long


def test_passes_filters_rejects_unparseable_duration():
    assert passes_filters(_details(duration_iso8601="garbage"), _settings()) is False


# ─── clean_queries / fallback_queries ─────────────────────────


def test_clean_queries_strips_dedupes_case_insensitively_and_bounds_length():
    assert clean_queries([" Cours Ampli Op ", "cours ampli op", "Méthode"], 2) == ["Cours Ampli Op", "Méthode"]


def test_clean_queries_truncates_very_long_queries():
    long_query = "x" * 500
    assert clean_queries([long_query], 5) == ["x" * 100]


def test_clean_queries_drops_blank_entries():
    assert clean_queries(["", "   ", "réel"], 5) == ["réel"]


def test_fallback_queries_are_deterministic_from_topic():
    assert fallback_queries("Les transformateurs") == [
        "Les transformateurs cours", "Les transformateurs exercices corrigés",
    ]


# ─── round_robin_merge ─────────────────────────────────────────


def _cand(vid: str) -> dict:
    return {"video_id": vid, "title": vid}


def test_round_robin_merge_alternates_between_queries():
    per_query = [[_cand("a1"), _cand("a2")], [_cand("b1"), _cand("b2")]]

    merged = round_robin_merge(per_query, limit=10)

    assert [c["video_id"] for c in merged] == ["a1", "b1", "a2", "b2"]


def test_round_robin_merge_deduplicates_across_queries():
    per_query = [[_cand("a1"), _cand("shared")], [_cand("shared"), _cand("b1")]]

    merged = round_robin_merge(per_query, limit=10)

    assert [c["video_id"] for c in merged] == ["a1", "shared", "b1"]


def test_round_robin_merge_stops_at_limit():
    per_query = [[_cand("a1"), _cand("a2")], [_cand("b1"), _cand("b2")]]

    assert [c["video_id"] for c in round_robin_merge(per_query, limit=2)] == ["a1", "b1"]


def test_round_robin_merge_handles_uneven_query_lengths():
    per_query = [[_cand("a1")], [_cand("b1"), _cand("b2"), _cand("b3")]]

    merged = round_robin_merge(per_query, limit=10)

    assert [c["video_id"] for c in merged] == ["a1", "b1", "b2", "b3"]


def test_round_robin_merge_of_no_queries_is_empty():
    assert round_robin_merge([], limit=5) == []


# ─── video_from_candidate ───────────────────────────────────────


def test_video_from_candidate_builds_a_full_course_video():
    video = video_from_candidate({
        "video_id": "v11111111vv", "title": "Titre", "channel_title": "Chaîne",
        "duration_seconds": 600, "published_at": "2024-01-01T00:00:00Z",
    })

    assert video is not None
    assert video.video_id == "v11111111vv"
    assert video.url == "https://www.youtube.com/watch?v=v11111111vv"
    assert video.channel == "Chaîne"
    assert video.duration_seconds == 600
    assert video.published_at == "2024-01-01T00:00:00Z"


def test_video_from_candidate_rejects_invalid_id():
    assert video_from_candidate({"video_id": "not-an-id", "title": "x"}) is None
