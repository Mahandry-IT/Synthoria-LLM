"""add course_video_notes table

Revision ID: 010
Revises: 009
Create Date: 2026-09-26
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import TIMESTAMP, UUID

# revision identifiers
revision = "010"
down_revision = "009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "course_video_notes",
        sa.Column(
            "session_id",
            UUID(as_uuid=True),
            sa.ForeignKey("course_sessions.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("video_id", sa.String(32), primary_key=True),
        sa.Column("note", sa.Text(), nullable=False, server_default=""),
        sa.Column("updated_at", TIMESTAMP(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("course_video_notes")
