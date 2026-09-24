"""Téléchargement sûr d'une image depuis un hôte en liste blanche (protection SSRF).

Jamais d'URL venant de Gemini ni d'une entrée utilisateur : uniquement celles renvoyées par les
API elles-mêmes (Commons, Openverse). La liste blanche est revalidée à chaque redirection ; toute
IP privée, loopback ou réservée est refusée ; la lecture est coupée en streaming dès que
`settings.media_max_bytes` est dépassé, sans jamais lire le corps en entier avant de vérifier la
taille. Le `Content-Type` déclaré n'est jamais cru sur parole — c'est `storage.normalize_image` qui
tranche via la signature réelle du fichier.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlsplit

import httpx

from app.core.config import Settings
from app.core.exceptions import MediaFetchRejected, MediaFetchTooLarge

ALLOWED_HOSTS = frozenset({"upload.wikimedia.org", "api.openverse.org"})
_MAX_REDIRECTS = 5


def _is_disallowed_ip(host: str) -> bool:
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False  # nom de domaine (pas une IP littérale) : résolu séparément
    return addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved or addr.is_multicast


def _assert_host_allowed(url: str) -> None:
    parts = urlsplit(url)
    if parts.scheme != "https":
        raise MediaFetchRejected(f"schéma refusé (https uniquement) : {parts.scheme!r}")
    host = parts.hostname or ""
    if host not in ALLOWED_HOSTS:
        raise MediaFetchRejected(f"hôte hors liste blanche : {host!r}")
    if _is_disallowed_ip(host):
        raise MediaFetchRejected(f"IP privée/loopback/réservée refusée : {host!r}")
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise MediaFetchRejected(f"résolution DNS impossible pour {host!r}") from exc
    for info in infos:
        ip = info[4][0]
        if _is_disallowed_ip(ip):
            raise MediaFetchRejected(f"{host!r} résout vers une IP privée/loopback/réservée ({ip})")


async def fetch_image(url: str, *, settings: Settings, client: httpx.AsyncClient | None = None) -> bytes:
    """Télécharge `url` en streaming. Lève `MediaFetchRejected`/`MediaFetchTooLarge`, jamais d'autre
    exception non-httpx (les erreurs réseau sont enveloppées dans `MediaFetchRejected`)."""
    _assert_host_allowed(url)
    owns_client = client is None
    http_client = client or httpx.AsyncClient(
        timeout=settings.media_web_timeout_seconds,
        headers={"User-Agent": settings.media_web_user_agent},
        follow_redirects=False,
    )
    try:
        current_url = url
        for _ in range(_MAX_REDIRECTS):
            async with http_client.stream("GET", current_url) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise MediaFetchRejected("redirection sans en-tête Location")
                    current_url = str(httpx.URL(current_url).join(location))
                    _assert_host_allowed(current_url)
                    continue
                response.raise_for_status()
                chunks: list[bytes] = []
                total = 0
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > settings.media_max_bytes:
                        raise MediaFetchTooLarge(f"corps > {settings.media_max_bytes} octets, coupé en streaming")
                    chunks.append(chunk)
                return b"".join(chunks)
        raise MediaFetchRejected(f"trop de redirections (> {_MAX_REDIRECTS})")
    except httpx.HTTPError as exc:
        raise MediaFetchRejected(f"téléchargement impossible : {exc}") from exc
    finally:
        if owns_client:
            await http_client.aclose()
