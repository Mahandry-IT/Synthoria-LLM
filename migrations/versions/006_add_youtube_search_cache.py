"""add youtube_search_cache table

Revision ID: 006
Revises: 005
Create Date: 2026-09-22
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP

# revision identifiers
revision = "006"
down_revision = "005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "youtube_search_cache",
        sa.Column("query_key", sa.String(64), primary_key=True),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("candidates", JSONB, nullable=False, server_default="[]"),
        sa.Column("fetched_at", TIMESTAMP(timezone=True), nullable=False),
    )
    op.create_index("idx_youtube_search_cache_fetched_at", "youtube_search_cache", ["fetched_at"])


def downgrade() -> None:
    op.drop_index("idx_youtube_search_cache_fetched_at", table_name="youtube_search_cache")
    op.drop_table("youtube_search_cache")
