"""Cache de réponses Gemini (`reformulate_query`/`describe_images`/`rank_images`) par hash de
requête — indépendant du cours qui a déclenché l'appel, pour que deux ingestions du même contenu
(ex. même PDF réingéré) partagent le cache.
"""

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import GeminiResponseCache


def compute_query_hash(method: str, *parts: str) -> str:
    """Clé de cache stable : méthode + parties de la requête (prompt, hash des images, nom du
    schema) — un hash borne la longueur de la clé primaire quel que soit le nombre d'images."""
    raw = "|".join((method, *parts))
    return hashlib.sha256(raw.encode()).hexdigest()


async def get_fresh(session: AsyncSession, query_hash: str, ttl: timedelta) -> dict[str, Any] | None:
    """Réponse en cache pour `query_hash`, ou None si absente ou périmée (ne la supprime pas)."""
    row = await session.get(GeminiResponseCache, query_hash)
    if row is None or row.created_at < datetime.now(timezone.utc) - ttl:
        return None
    return row.response


async def upsert(session: AsyncSession, query_hash: str, method: str, response: dict[str, Any]) -> None:
    """Enregistre (ou remplace) la réponse d'une requête, horodatée à maintenant."""
    row = await session.get(GeminiResponseCache, query_hash)
    now = datetime.now(timezone.utc)
    if row is None:
        session.add(GeminiResponseCache(query_hash=query_hash, method=method, response=response, created_at=now))
    else:
        row.method, row.response, row.created_at = method, response, now
    await session.commit()


async def purge_older_than(session: AsyncSession, ttl: timedelta) -> int:
    """Supprime les entrées plus vieilles que `ttl` ; retourne le nombre de lignes supprimées."""
    cutoff = datetime.now(timezone.utc) - ttl
    result = await session.execute(delete(GeminiResponseCache).where(GeminiResponseCache.created_at < cutoff))
    await session.commit()
    return result.rowcount or 0
