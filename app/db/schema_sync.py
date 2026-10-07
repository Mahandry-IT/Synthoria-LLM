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
    # migration 017_add_course_chat_message_versions
    "ALTER TABLE course_chat_messages ADD COLUMN IF NOT EXISTS parent_id UUID "
    "REFERENCES course_chat_messages(id) ON DELETE CASCADE",
    "ALTER TABLE course_chat_messages ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ",
    "CREATE INDEX IF NOT EXISTS idx_course_chat_messages_parent ON course_chat_messages (parent_id)",
)

# Rattrapages de données idempotents, joués après les colonnes.
DATA_BACKFILLS: tuple[str, ...] = (
    # migration 017 : chats antérieurs (aucun parent renseigné) chaînés par ordre chronologique.
    """
    WITH chained AS (
        SELECT id, LAG(id) OVER (PARTITION BY session_id ORDER BY created_at, role DESC) AS prev_id
        FROM course_chat_messages
        WHERE session_id IN (
            SELECT session_id FROM course_chat_messages
            GROUP BY session_id
            HAVING COUNT(*) > 1 AND COUNT(parent_id) = 0
        )
    )
    UPDATE course_chat_messages AS m
    SET parent_id = chained.prev_id
    FROM chained
    WHERE m.id = chained.id AND chained.prev_id IS NOT NULL
    """,
)


async def sync_added_columns(conn: AsyncConnection) -> None:
    """Ajoute les colonnes manquantes puis joue les rattrapages (no-op si déjà appliqués)."""
    for statement in (*ADDED_COLUMNS, *DATA_BACKFILLS):
        await conn.execute(text(statement))
