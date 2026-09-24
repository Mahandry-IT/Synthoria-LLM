"""Cache des recherches d'images web (Wikimedia Commons / Openverse), par requête normalisée.

Un résultat négatif (aucun candidat retenu) est mis en cache au même titre qu'un résultat positif :
`asset_id=None` évite de retenter une recherche qui a déjà échoué avant `MEDIA_WEB_CACHE_TTL_HOURS`.
"""

import hashlib
import re
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import MediaQueryCache

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_query_key(query: str) -> str:
    """Clé de cache stable : minuscules, espaces compactés. Un hash borne la longueur de la clé."""
    normalized = _WHITESPACE_RE.sub(" ", query.strip().casefold())
    return hashlib.sha256(normalized.encode()).hexdigest()


class CacheHit:
    """Résultat d'un hit de cache : `found` distingue « pas en cache » de « en cache, négatif »."""

    __slots__ = ("found", "asset_id")

    def __init__(self, *, found: bool, asset_id: uuid.UUID | None) -> None:
        self.found = found
        self.asset_id = asset_id


async def get_fresh(session: AsyncSession, query_hash: str, ttl: timedelta) -> CacheHit:
    """`CacheHit(found=False, ...)` si absent ou périmé (ne le supprime pas) ; sinon le hit (positif ou négatif)."""
    row = await session.get(MediaQueryCache, query_hash)
    if row is None or row.created_at < datetime.now(timezone.utc) - ttl:
        return CacheHit(found=False, asset_id=None)
    return CacheHit(found=True, asset_id=row.asset_id)


async def upsert(session: AsyncSession, query_hash: str, query: str, asset_id: uuid.UUID | None) -> None:
    """Enregistre (ou remplace) le résultat d'une requête, horodaté à maintenant. `asset_id=None` = négatif."""
    row = await session.get(MediaQueryCache, query_hash)
    now = datetime.now(timezone.utc)
    if row is None:
        session.add(MediaQueryCache(query_hash=query_hash, query=query, asset_id=asset_id, created_at=now))
    else:
        row.query, row.asset_id, row.created_at = query, asset_id, now
    await session.commit()


async def purge_older_than(session: AsyncSession, ttl: timedelta) -> int:
    """Supprime les entrées plus vieilles que `ttl` ; retourne le nombre de lignes supprimées."""
    cutoff = datetime.now(timezone.utc) - ttl
    result = await session.execute(delete(MediaQueryCache).where(MediaQueryCache.created_at < cutoff))
    await session.commit()
    return result.rowcount or 0
