"""add course_chat_messages table

Revision ID: 016
Revises: 015
Create Date: 2026-10-04
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, UUID

# revision identifiers
revision = "016"
down_revision = "015"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "course_chat_messages",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "session_id",
            UUID(as_uuid=True),
            sa.ForeignKey("course_sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="answered"),
        sa.Column("sources", JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("created_at", TIMESTAMP(timezone=True), nullable=False),
    )
    op.create_index(
        "idx_course_chat_messages_session_created",
        "course_chat_messages",
        ["session_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("idx_course_chat_messages_session_created", table_name="course_chat_messages")
    op.drop_table("course_chat_messages")
