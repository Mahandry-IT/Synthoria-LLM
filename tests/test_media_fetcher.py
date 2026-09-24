import httpx
import pytest

from app.core.config import Settings
from app.core.exceptions import MediaFetchRejected, MediaFetchTooLarge
from app.services.media.fetcher import fetch_image


def _settings(**overrides) -> Settings:
    return Settings(media_web_user_agent="test-agent/1.0 (test)", media_max_bytes=1000, **overrides)


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False)


@pytest.mark.asyncio
async def test_rejects_host_outside_allowlist():
    with pytest.raises(MediaFetchRejected, match="hôte hors liste blanche"):
        await fetch_image("https://evil.example.com/image.jpg", settings=_settings())


@pytest.mark.asyncio
async def test_rejects_non_https_scheme():
    with pytest.raises(MediaFetchRejected, match="schéma refusé"):
        await fetch_image("http://upload.wikimedia.org/image.jpg", settings=_settings())


@pytest.mark.asyncio
async def test_rejects_literal_loopback_ip():
    """Une IP littérale n'est de toute façon jamais dans la liste blanche (hôtes nommés)."""
    with pytest.raises(MediaFetchRejected, match="hôte hors liste blanche"):
        await fetch_image("https://127.0.0.1/image.jpg", settings=_settings())


@pytest.mark.asyncio
async def test_rejects_allowed_hostname_resolving_to_private_ip(monkeypatch):
    """Un hôte en liste blanche qui résoudrait (DNS rebinding) vers une IP privée est refusé."""
    import socket

    def fake_getaddrinfo(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))]

    monkeypatch.setattr("app.services.media.fetcher.socket.getaddrinfo", fake_getaddrinfo)

    with pytest.raises(MediaFetchRejected, match="IP privée"):
        await fetch_image("https://upload.wikimedia.org/image.jpg", settings=_settings())


@pytest.mark.asyncio
async def test_downloads_from_allowed_host():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"fake-image-bytes")

    client = _client(handler)
    result = await fetch_image("https://upload.wikimedia.org/image.jpg", settings=_settings(), client=client)

    assert result == b"fake-image-bytes"
    await client.aclose()


@pytest.mark.asyncio
async def test_follows_redirect_to_allowed_host():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if "upload.wikimedia.org" in str(request.url) and len(calls) == 1:
            return httpx.Response(302, headers={"Location": "https://upload.wikimedia.org/final.jpg"})
        return httpx.Response(200, content=b"final-bytes")

    client = _client(handler)
    result = await fetch_image("https://upload.wikimedia.org/redirect.jpg", settings=_settings(), client=client)

    assert result == b"final-bytes"
    assert len(calls) == 2
    await client.aclose()


@pytest.mark.asyncio
async def test_rejects_redirect_to_disallowed_host():
    def handler(request: httpx.Request) -> httpx.Response:
        if "upload.wikimedia.org" in str(request.url):
            return httpx.Response(302, headers={"Location": "https://attacker.example.com/steal"})
        return httpx.Response(200, content=b"should-not-reach")

    client = _client(handler)
    with pytest.raises(MediaFetchRejected, match="hôte hors liste blanche"):
        await fetch_image("https://upload.wikimedia.org/redirect.jpg", settings=_settings(), client=client)
    await client.aclose()


@pytest.mark.asyncio
async def test_cuts_body_exceeding_max_bytes_without_reading_it_all():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 5000)  # > media_max_bytes=1000

    client = _client(handler)
    with pytest.raises(MediaFetchTooLarge):
        await fetch_image("https://upload.wikimedia.org/huge.jpg", settings=_settings(), client=client)
    await client.aclose()


@pytest.mark.asyncio
async def test_wraps_network_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    client = _client(handler)
    with pytest.raises(MediaFetchRejected):
        await fetch_image("https://upload.wikimedia.org/image.jpg", settings=_settings(), client=client)
    await client.aclose()
