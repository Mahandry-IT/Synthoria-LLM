import httpx
import pytest

from app.core.exceptions import YoutubeConfigError, YoutubeQuotaExceeded, YoutubeUnavailable
from app.services.youtube_data_client import YoutubeDataClient, YoutubeVideoDetails


def _client(handler) -> YoutubeDataClient:
    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport, base_url="https://www.googleapis.com/youtube/v3")
    return YoutubeDataClient("fake-key", client=http_client)


def _search_response(ids: list[str]) -> httpx.Response:
    return httpx.Response(200, json={"items": [{"id": {"videoId": i}} for i in ids]})


def _error_response(status: int, reason: str) -> httpx.Response:
    return httpx.Response(status, json={"error": {"code": status, "errors": [{"reason": reason}]}})


@pytest.mark.asyncio
async def test_api_key_travels_in_header_never_in_the_url():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["header"] = request.headers.get("x-goog-api-key")
        seen["url"] = str(request.url)
        return _search_response(["abc"])

    client = _client(handler)

    await client.search("transformateur électrique")

    assert seen["header"] == "fake-key"
    assert "fake-key" not in seen["url"]
    assert "key=" not in seen["url"]


@pytest.mark.asyncio
async def test_search_returns_video_ids_in_order():
    client = _client(lambda r: _search_response(["a11111111aa", "b22222222bb"]))

    ids = await client.search("cours de thermodynamique", max_results=8)

    assert ids == ["a11111111aa", "b22222222bb"]


@pytest.mark.asyncio
async def test_search_ignores_items_without_a_video_id():
    client = _client(lambda r: httpx.Response(200, json={"items": [{"id": {}}, {"id": {"videoId": "x11111111xx"}}]}))

    assert await client.search("q") == ["x11111111xx"]


@pytest.mark.asyncio
async def test_details_maps_snippet_content_details_and_status():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "items": [{
                "id": "v11111111vv",
                "snippet": {
                    "title": "Titre", "channelTitle": "Chaîne", "description": "Desc",
                    "publishedAt": "2024-01-01T00:00:00Z", "liveBroadcastContent": "none",
                },
                "contentDetails": {"duration": "PT12M34S"},
                "status": {"embeddable": True, "privacyStatus": "public"},
            }]
        })

    client = _client(handler)

    details = await client.details(["v11111111vv"])

    assert details == [
        YoutubeVideoDetails(
            video_id="v11111111vv", title="Titre", channel_title="Chaîne", description="Desc",
            published_at="2024-01-01T00:00:00Z", duration_iso8601="PT12M34S",
            embeddable=True, privacy_status="public", live_broadcast_content="none",
        )
    ]


@pytest.mark.asyncio
async def test_details_of_empty_list_makes_no_request():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"items": []})

    client = _client(handler)

    assert await client.details([]) == []
    assert calls == []


@pytest.mark.asyncio
async def test_several_queries_worth_of_ids_still_use_a_single_details_call():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path.endswith("/videos"):
            ids = request.url.params["id"].split(",")
            return httpx.Response(200, json={"items": [{"id": i, "snippet": {}, "contentDetails": {}, "status": {}} for i in ids]})
        return _search_response(["id"])

    client = _client(handler)

    await client.details(["id1", "id2", "id3"])

    video_calls = [c for c in calls if c.url.path.endswith("/videos")]
    assert len(video_calls) == 1
    assert video_calls[0].url.params["id"] == "id1,id2,id3"


@pytest.mark.asyncio
async def test_details_truncates_to_fifty_ids():
    seen_ids = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_ids["ids"] = request.url.params["id"].split(",")
        return httpx.Response(200, json={"items": []})

    client = _client(handler)

    await client.details([f"id{i:040d}" for i in range(60)])

    assert len(seen_ids["ids"]) == 50


@pytest.mark.asyncio
async def test_quota_exceeded_maps_to_youtube_quota_exceeded():
    client = _client(lambda r: _error_response(403, "quotaExceeded"))

    with pytest.raises(YoutubeQuotaExceeded):
        await client.search("q")


@pytest.mark.asyncio
async def test_daily_limit_exceeded_also_maps_to_quota_exceeded():
    client = _client(lambda r: _error_response(403, "dailyLimitExceeded"))

    with pytest.raises(YoutubeQuotaExceeded):
        await client.search("q")


@pytest.mark.asyncio
async def test_invalid_key_maps_to_config_error():
    client = _client(lambda r: _error_response(403, "keyInvalid"))

    with pytest.raises(YoutubeConfigError):
        await client.search("q")


@pytest.mark.asyncio
async def test_bad_request_maps_to_config_error():
    client = _client(lambda r: httpx.Response(400, json={"error": {"code": 400, "errors": []}}))

    with pytest.raises(YoutubeConfigError):
        await client.search("q")


@pytest.mark.asyncio
async def test_server_error_maps_to_unavailable():
    client = _client(lambda r: httpx.Response(503, text="oops"))

    with pytest.raises(YoutubeUnavailable):
        await client.search("q")


@pytest.mark.asyncio
async def test_timeout_maps_to_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out")

    client = _client(handler)

    with pytest.raises(YoutubeUnavailable):
        await client.search("q")


@pytest.mark.asyncio
async def test_network_error_maps_to_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    client = _client(handler)

    with pytest.raises(YoutubeUnavailable):
        await client.details(["id"])
