"""add flashcard_reviews table

Revision ID: 005
Revises: 004
Create Date: 2026-09-22
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import TIMESTAMP, UUID

# revision identifiers
revision = "005"
down_revision = "004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "flashcard_reviews",
        sa.Column(
            "session_id",
            UUID(as_uuid=True),
            sa.ForeignKey("course_sessions.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("card_id", sa.String(64), primary_key=True),
        sa.Column("box", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("due_at", TIMESTAMP(timezone=True), nullable=False),
        sa.Column("last_result", sa.String(10), nullable=True),
    )
    op.create_index("idx_flashcard_reviews_due_at", "flashcard_reviews", ["due_at"])


def downgrade() -> None:
    op.drop_index("idx_flashcard_reviews_due_at", table_name="flashcard_reviews")
    op.drop_table("flashcard_reviews")
