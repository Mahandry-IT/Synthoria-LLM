"""add depth column to course_plans

Revision ID: 015
Revises: 014
Create Date: 2026-10-02

Idempotent (`IF NOT EXISTS`) : la même colonne est aussi ajoutée au démarrage de l'API
(`app.db.schema_sync`) pour les bases créées par `create_all`, sans table `alembic_version`.
"""
from alembic import op

# revision identifiers
revision = "015"
down_revision = "014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE course_plans ADD COLUMN IF NOT EXISTS depth VARCHAR(12) NOT NULL DEFAULT 'approfondi'"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE course_plans DROP COLUMN IF EXISTS depth")
