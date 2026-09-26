"""add folder/subfolder columns to course_sessions

Revision ID: 011
Revises: 010
Create Date: 2026-09-26
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = "011"
down_revision = "010"
branch_labels = None
depends_on = None

DEFAULT_FOLDER = "Général"
DEFAULT_SUBFOLDER = "Non classé"


def upgrade() -> None:
    op.add_column(
        "course_sessions",
        sa.Column("folder", sa.String(200), nullable=False, server_default=DEFAULT_FOLDER),
    )
    op.add_column(
        "course_sessions",
        sa.Column("subfolder", sa.String(200), nullable=False, server_default=DEFAULT_SUBFOLDER),
    )
    op.create_index("idx_course_sessions_folder", "course_sessions", ["folder", "subfolder"])


def downgrade() -> None:
    op.drop_index("idx_course_sessions_folder", table_name="course_sessions")
    op.drop_column("course_sessions", "subfolder")
    op.drop_column("course_sessions", "folder")
