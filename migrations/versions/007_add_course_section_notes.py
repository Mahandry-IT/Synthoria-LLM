"""add course_section_notes table

Revision ID: 007
Revises: 006
Create Date: 2026-09-23
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import TIMESTAMP, UUID

# revision identifiers
revision = "007"
down_revision = "006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "course_section_notes",
        sa.Column(
            "session_id",
            UUID(as_uuid=True),
            sa.ForeignKey("course_sessions.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("section_id", sa.String(64), primary_key=True),
        sa.Column("note", sa.Text(), nullable=False, server_default=""),
        sa.Column("updated_at", TIMESTAMP(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("course_section_notes")
