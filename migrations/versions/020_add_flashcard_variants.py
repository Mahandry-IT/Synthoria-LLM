"""add flashcard_variants table and flashcard_reviews.variant_no

Revision ID: 020
Revises: 019
Create Date: 2026-10-09

La colonne est ajoutée avec `IF NOT EXISTS` : la même instruction est jouée au démarrage de l'API
(`app.db.schema_sync`) pour les bases créées par `create_all`, sans table `alembic_version`.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, UUID

# revision identifiers
revision = "020"
down_revision = "019"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "flashcard_variants",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "session_id", UUID(as_uuid=True),
            sa.ForeignKey("course_sessions.id", ondelete="CASCADE"), nullable=False,
        ),
        sa.Column("card_id", sa.String(64), nullable=False),
        sa.Column("variant_no", sa.Integer, nullable=False),
        sa.Column("front", sa.Text, nullable=False),
        sa.Column("choices", JSONB, nullable=False, server_default="[]"),
        sa.Column("correct_indices", JSONB, nullable=False, server_default="[]"),
        sa.Column("explanation", sa.Text, nullable=False, server_default=""),
        sa.Column("created_at", TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("session_id", "card_id", "variant_no", name="uq_flashcard_variants_card_variant"),
    )
    op.execute("ALTER TABLE flashcard_reviews ADD COLUMN IF NOT EXISTS variant_no INTEGER NOT NULL DEFAULT 0")


def downgrade() -> None:
    op.drop_column("flashcard_reviews", "variant_no")
    op.drop_table("flashcard_variants")
