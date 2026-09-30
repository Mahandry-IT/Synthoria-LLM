"""État de quota Gemini partagé entre les process `api` et `worker` (chacun a son propre
`GeminiRateLimiter` en mémoire, non coordonné — voir `app/services/gemini_quota_manager.py`).

Le compteur `request_count` est incrémenté concurremment par les deux process : contrairement
aux caches du reste du repo (où lire-puis-écrire est acceptable, un écrasement concurrent y étant
inoffensif), c'est ici une vraie race condition — toute écriture passe par une requête SQL
atomique (`UPDATE ... RETURNING`), jamais par `session.get()` suivi d'une mutation.
"""

from datetime import date, datetime, timezone

from sqlalchemy import case, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import GeminiModelQuota


def _increment_statement(model: str, today: date, now: datetime):
    """`UPDATE ... RETURNING` unique gérant à la fois l'incrément (jour courant) et la
    réinitialisation (ligne existante mais périmée) — pas de filtre sur `day` en clause `WHERE`,
    pour trouver la ligne quel que soit le jour qu'elle porte encore."""
    return (
        update(GeminiModelQuota)
        .where(GeminiModelQuota.model == model)
        .values(
            day=today,
            request_count=case(
                (GeminiModelQuota.day == today, GeminiModelQuota.request_count + 1),
                else_=1,
            ),
            updated_at=now,
        )
        .returning(GeminiModelQuota.request_count)
    )


async def increment(session: AsyncSession, model: str, today: date) -> int:
    """Incrémente le compteur du jour pour `model` (le réinitialise à 1 si la ligne existante
    porte un jour périmé). Retourne le compteur après incrément.
    """
    now = datetime.now(timezone.utc)
    result = await session.execute(_increment_statement(model, today, now))
    row = result.first()
    if row is not None:
        await session.commit()
        return row[0]

    # Ligne absente : premier appel jamais vu pour ce modèle.
    try:
        session.add(GeminiModelQuota(model=model, day=today, request_count=1, updated_at=now))
        await session.commit()
        return 1
    except IntegrityError:
        # Un autre process (api/worker) vient d'insérer la ligne entre l'UPDATE et l'INSERT :
        # elle existe désormais, le même UPDATE réussit forcément cette fois.
        await session.rollback()
        result = await session.execute(_increment_statement(model, today, now))
        await session.commit()
        return result.scalar_one()


async def mark_exhausted(session: AsyncSession, model: str, exhausted_until: datetime) -> None:
    """Marque `model` comme épuisé jusqu'à `exhausted_until` (lecture-écriture acceptable ici :
    un dépassement mutuel entre process ne fait qu'écraser la même information, sans compteur)."""
    row = await session.get(GeminiModelQuota, model)
    now = datetime.now(timezone.utc)
    if row is None:
        session.add(
            GeminiModelQuota(
                model=model, day=now.date(), request_count=0,
                exhausted_until=exhausted_until, updated_at=now,
            )
        )
    else:
        row.exhausted_until = exhausted_until
        row.updated_at = now
    await session.commit()


async def get_states(session: AsyncSession, models: list[str], today: date) -> dict[str, GeminiModelQuota]:
    """État actuel des modèles demandés, indexé par nom. Un modèle absent de la table
    (jamais appelé) n'apparaît pas dans le résultat — à l'appelant de traiter cette absence."""
    if not models:
        return {}
    result = await session.execute(select(GeminiModelQuota).where(GeminiModelQuota.model.in_(models)))
    return {row.model: row for row in result.scalars().all()}
