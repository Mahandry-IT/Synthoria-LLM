"""add gemini_model_quota table

Revision ID: 013
Revises: 012
Create Date: 2026-09-30
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import TIMESTAMP

# revision identifiers
revision = "013"
down_revision = "012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "gemini_model_quota",
        sa.Column("model", sa.String(100), primary_key=True),
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column("request_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("exhausted_until", TIMESTAMP(timezone=True), nullable=True),
        sa.Column("updated_at", TIMESTAMP(timezone=True), nullable=False),
    )
    op.create_index("idx_gemini_model_quota_day", "gemini_model_quota", ["day"])


def downgrade() -> None:
    op.drop_index("idx_gemini_model_quota_day", table_name="gemini_model_quota")
    op.drop_table("gemini_model_quota")
