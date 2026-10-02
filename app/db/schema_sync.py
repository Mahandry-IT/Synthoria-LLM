"""Colonnes ajoutées après coup à des tables existantes, appliquées au démarrage de l'API.

`Base.metadata.create_all` crée les tables manquantes mais n'ajoute jamais de colonne à une table
déjà présente : sans ce filet, une base créée avant une migration Alembic (aucune table
`alembic_version`, migrations non jouées par le conteneur) casserait toute lecture du modèle.
Chaque instruction est idempotente et reprend à l'identique la migration correspondante.
"""

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

ADDED_COLUMNS: tuple[str, ...] = (
    # migration 015_add_course_plan_depth
    "ALTER TABLE course_plans ADD COLUMN IF NOT EXISTS depth VARCHAR(12) NOT NULL DEFAULT 'approfondi'",
)


async def sync_added_columns(conn: AsyncConnection) -> None:
    """Ajoute les colonnes manquantes (no-op si déjà présentes)."""
    for statement in ADDED_COLUMNS:
        await conn.execute(text(statement))
