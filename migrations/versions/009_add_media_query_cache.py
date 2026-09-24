"""add media_query_cache table

Revision ID: 009
Revises: 008
Create Date: 2026-09-24
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import TIMESTAMP, UUID

# revision identifiers
revision = "009"
down_revision = "008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "media_query_cache",
        sa.Column("query_hash", sa.String(64), primary_key=True),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column(
            "asset_id",
            UUID(as_uuid=True),
            sa.ForeignKey("media_assets.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("created_at", TIMESTAMP(timezone=True), nullable=False),
    )
    op.create_index("idx_media_query_cache_created_at", "media_query_cache", ["created_at"])


def downgrade() -> None:
    op.drop_index("idx_media_query_cache_created_at", table_name="media_query_cache")
    op.drop_table("media_query_cache")
