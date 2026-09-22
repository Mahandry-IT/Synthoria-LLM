"""Cache des résultats de recherche YouTube (`search.list` + `videos.list` enrichis).

Le cache est partagé entre cours : la clé dérive de la requête normalisée (+ langue, région),
jamais du cours qui l'a déclenchée.
"""

import hashlib
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import YoutubeSearchCache

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_query_key(query: str, *, language: str, region: str) -> str:
    """Clé de cache stable : minuscules, espaces compactés (accents conservés), + langue/région.

    Un hash (plutôt que la requête brute) borne la longueur de la clé primaire quelle que soit
    la longueur de la requête, et évite les soucis de casse de colonne selon la locale DB.
    """
    normalized = _WHITESPACE_RE.sub(" ", query.strip().casefold())
    return hashlib.sha256(f"{normalized}|{language}|{region}".encode()).hexdigest()


async def get_fresh(session: AsyncSession, query_key: str, ttl: timedelta) -> list[dict[str, Any]] | None:
    """Candidats en cache pour `query_key`, ou None si absents ou périmés (ne les supprime pas)."""
    row = await session.get(YoutubeSearchCache, query_key)
    if row is None or row.fetched_at < datetime.now(timezone.utc) - ttl:
        return None
    return row.candidates


async def upsert(session: AsyncSession, query_key: str, query: str, candidates: list[dict[str, Any]]) -> None:
    """Enregistre (ou remplace) les candidats d'une requête, horodatés à maintenant."""
    row = await session.get(YoutubeSearchCache, query_key)
    now = datetime.now(timezone.utc)
    if row is None:
        session.add(YoutubeSearchCache(query_key=query_key, query=query, candidates=candidates, fetched_at=now))
    else:
        row.query, row.candidates, row.fetched_at = query, candidates, now
    await session.commit()


async def purge_older_than(session: AsyncSession, ttl: timedelta) -> int:
    """Supprime les entrées plus vieilles que `ttl` ; retourne le nombre de lignes supprimées."""
    cutoff = datetime.now(timezone.utc) - ttl
    result = await session.execute(delete(YoutubeSearchCache).where(YoutubeSearchCache.fetched_at < cutoff))
    await session.commit()
    return result.rowcount or 0
