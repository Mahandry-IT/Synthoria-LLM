"""add parent_id and deleted_at to course_chat_messages (message versions and deletion)

Revision ID: 017
Revises: 016
Create Date: 2026-10-07

Idempotent (`IF NOT EXISTS`, rattrapage limité aux chats jamais chaînés) : les mêmes instructions
sont aussi jouées au démarrage de l'API (`app.db.schema_sync`) pour les bases créées par
`create_all`, sans table `alembic_version`.
"""
from alembic import op

# revision identifiers
revision = "017"
down_revision = "016"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE course_chat_messages ADD COLUMN IF NOT EXISTS parent_id UUID "
        "REFERENCES course_chat_messages(id) ON DELETE CASCADE"
    )
    op.execute("ALTER TABLE course_chat_messages ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ")
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_course_chat_messages_parent ON course_chat_messages (parent_id)"
    )
    # Rattrapage : les chats antérieurs (aucun parent renseigné) deviennent une branche unique,
    # chaque message ayant pour parent le précédent (question avant réponse à horodatage égal).
    op.execute(
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
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_course_chat_messages_parent")
    op.execute("ALTER TABLE course_chat_messages DROP COLUMN IF EXISTS deleted_at")
    op.execute("ALTER TABLE course_chat_messages DROP COLUMN IF EXISTS parent_id")
