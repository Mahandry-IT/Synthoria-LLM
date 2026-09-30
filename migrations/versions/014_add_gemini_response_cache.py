"""add gemini_response_cache table

Revision ID: 014
Revises: 013
Create Date: 2026-09-30
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP

# revision identifiers
revision = "014"
down_revision = "013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "gemini_response_cache",
        sa.Column("query_hash", sa.String(64), primary_key=True),
        sa.Column("method", sa.String(50), nullable=False),
        sa.Column("response", JSONB, nullable=False),
        sa.Column("created_at", TIMESTAMP(timezone=True), nullable=False),
    )
    op.create_index("idx_gemini_response_cache_created_at", "gemini_response_cache", ["created_at"])


def downgrade() -> None:
    op.drop_index("idx_gemini_response_cache_created_at", table_name="gemini_response_cache")
    op.drop_table("gemini_response_cache")
