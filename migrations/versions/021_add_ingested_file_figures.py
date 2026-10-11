"""add ingested_file_figures table

Revision ID: 021
Revises: 020
Create Date: 2026-10-11

Figures des PDF ingérés, réutilisables par les blocs IMAGE `image_source="pdf"`. Les bases créées
par `create_all` (sans table `alembic_version`) reçoivent cette table au démarrage de l'API.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import TIMESTAMP, UUID

# revision identifiers
revision = "021"
down_revision = "020"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ingested_file_figures",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("filename", sa.String(512), nullable=False),
        sa.Column("ref", sa.String(32), nullable=False),
        sa.Column("page", sa.Integer, nullable=False),
        sa.Column(
            "asset_id", UUID(as_uuid=True),
            sa.ForeignKey("media_assets.id", ondelete="CASCADE"), nullable=False,
        ),
        sa.Column("created_at", TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("filename", "ref", name="uq_ingested_file_figures_filename_ref"),
    )
    op.create_index("idx_ingested_file_figures_ref", "ingested_file_figures", ["ref"])


def downgrade() -> None:
    op.drop_index("idx_ingested_file_figures_ref", table_name="ingested_file_figures")
    op.drop_table("ingested_file_figures")
