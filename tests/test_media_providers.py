import httpx
import pytest

from app.core.exceptions import MediaWebError
from app.services.media.providers.openverse import OpenverseProvider
from app.services.media.providers.wikimedia import WikimediaProvider


def _wikimedia_response(pages: dict) -> httpx.Response:
    return httpx.Response(200, json={"query": {"pages": pages}})


def _wikimedia_page(
    title="File:Example.jpg", mime="image/jpeg", thumburl="https://upload.wikimedia.org/thumb/example_1600px.jpg",
    width=3000, height=2000, thumbwidth=1600, thumbheight=1067,
    license_name="CC BY-SA 4.0", artist='<a href="https://example.org">Jane Doe</a>', license_url="https://creativecommons.org/licenses/by-sa/4.0",
) -> dict:
    return {
        "title": title,
        "imageinfo": [
            {
                "mime": mime,
                "thumburl": thumburl,
                "thumbwidth": thumbwidth,
                "thumbheight": thumbheight,
                "width": width,
                "height": height,
                "descriptionurl": f"https://commons.wikimedia.org/wiki/{title}",
                "extmetadata": {
                    "LicenseShortName": {"value": license_name},
                    "Artist": {"value": artist},
                    "LicenseUrl": {"value": license_url},
                },
            }
        ],
    }


@pytest.mark.asyncio
async def test_wikimedia_parses_results_into_candidates():
    def handler(request: httpx.Request) -> httpx.Response:
        return _wikimedia_response({"1": _wikimedia_page()})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = WikimediaProvider(user_agent="test-agent/1.0", client=client)

    candidates = await provider.search("chat noir", limit=5)

    assert len(candidates) == 1
    c = candidates[0]
    assert c.provider == "wikimedia"
    assert c.download_url == c.thumb_url == "https://upload.wikimedia.org/thumb/example_1600px.jpg"
    assert c.width == 1600 and c.height == 1067  # dimensions de la miniature, pas de l'original
    assert c.license == "CC BY-SA 4.0"
    assert c.author == '<a href="https://example.org">Jane Doe</a>'  # nettoyé plus tard par licenses.strip_html
    await client.aclose()


@pytest.mark.asyncio
async def test_wikimedia_excludes_pages_without_thumburl():
    page = _wikimedia_page()
    del page["imageinfo"][0]["thumburl"]

    def handler(request: httpx.Request) -> httpx.Response:
        return _wikimedia_response({"1": page})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = WikimediaProvider(user_agent="test-agent/1.0", client=client)

    assert await provider.search("q", limit=5) == []
    await client.aclose()


@pytest.mark.asyncio
async def test_wikimedia_excludes_non_image_mime():
    page = _wikimedia_page(mime="application/pdf")

    def handler(request: httpx.Request) -> httpx.Response:
        return _wikimedia_response({"1": page})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = WikimediaProvider(user_agent="test-agent/1.0", client=client)

    assert await provider.search("q", limit=5) == []
    await client.aclose()


@pytest.mark.asyncio
async def test_wikimedia_skips_pages_without_imageinfo():
    def handler(request: httpx.Request) -> httpx.Response:
        return _wikimedia_response({"1": {"title": "File:NoInfo.jpg"}, "2": _wikimedia_page(title="File:Ok.jpg")})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = WikimediaProvider(user_agent="test-agent/1.0", client=client)

    candidates = await provider.search("q", limit=5)
    assert len(candidates) == 1 and candidates[0].title == "File:Ok.jpg"
    await client.aclose()


@pytest.mark.asyncio
async def test_wikimedia_wraps_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = WikimediaProvider(user_agent="test-agent/1.0", client=client)

    with pytest.raises(MediaWebError):
        await provider.search("q", limit=5)
    await client.aclose()


def _openverse_item(**overrides) -> dict:
    item = {
        "title": "A black cat",
        "creator": "Jane Doe",
        "url": "https://flickr.com/original.jpg",  # jamais utilisée : un seul hôte en liste blanche
        "thumbnail": "https://api.openverse.org/v1/images/abc/thumb/",
        "foreign_landing_url": "https://flickr.com/photos/abc",
        "license": "by-sa",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "width": 1200,
        "height": 800,
    }
    item.update(overrides)
    return item


@pytest.mark.asyncio
async def test_openverse_never_returns_the_origin_host_url():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [_openverse_item()]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://api.openverse.org/v1")
    provider = OpenverseProvider(user_agent="test-agent/1.0", client=client)

    candidates = await provider.search("chat noir", limit=5)

    assert len(candidates) == 1
    assert candidates[0].download_url == "https://api.openverse.org/v1/images/abc/thumb/"
    assert "flickr" not in candidates[0].download_url
    await client.aclose()


@pytest.mark.asyncio
async def test_openverse_excludes_items_without_thumbnail():
    item = _openverse_item()
    del item["thumbnail"]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [item]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://api.openverse.org/v1")
    provider = OpenverseProvider(user_agent="test-agent/1.0", client=client)

    assert await provider.search("q", limit=5) == []
    await client.aclose()


@pytest.mark.asyncio
async def test_openverse_anonymous_when_no_credentials():
    seen_auth = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_auth.append(request.headers.get("authorization"))
        return httpx.Response(200, json={"results": []})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://api.openverse.org/v1")
    provider = OpenverseProvider(user_agent="test-agent/1.0", client=client)

    await provider.search("q", limit=5)

    assert seen_auth == [None]
    await client.aclose()


@pytest.mark.asyncio
async def test_openverse_fetches_and_reuses_bearer_token():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.url.path.endswith("/auth_tokens/token/"):
            return httpx.Response(200, json={"access_token": "tok-123", "expires_in": 3600})
        return httpx.Response(200, json={"results": []})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://api.openverse.org/v1")
    provider = OpenverseProvider(user_agent="test-agent/1.0", client_id="id", client_secret="secret", client=client)

    await provider.search("q1", limit=5)
    await provider.search("q2", limit=5)

    token_calls = [c for c in calls if "auth_tokens" in c]
    assert len(token_calls) == 1  # jeton récupéré une fois, réutilisé pour la 2e recherche
    await client.aclose()


@pytest.mark.asyncio
async def test_openverse_falls_back_to_anonymous_when_auth_fails():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/auth_tokens/token/"):
            return httpx.Response(401)
        return httpx.Response(200, json={"results": []})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://api.openverse.org/v1")
    provider = OpenverseProvider(user_agent="test-agent/1.0", client_id="id", client_secret="secret", client=client)

    candidates = await provider.search("q", limit=5)  # ne lève pas, malgré l'échec d'auth
    assert candidates == []
    await client.aclose()


@pytest.mark.asyncio
async def test_openverse_wraps_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://api.openverse.org/v1")
    provider = OpenverseProvider(user_agent="test-agent/1.0", client=client)

    with pytest.raises(MediaWebError):
        await provider.search("q", limit=5)
    await client.aclose()
