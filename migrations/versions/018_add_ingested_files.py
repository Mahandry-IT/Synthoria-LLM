"""add ingested_files table (folder/subfolder of files ingested in the vector store)

Revision ID: 018
Revises: 017
Create Date: 2026-10-07
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import TIMESTAMP

# revision identifiers
revision = "018"
down_revision = "017"
branch_labels = None
depends_on = None

DEFAULT_FOLDER = "Général"
DEFAULT_SUBFOLDER = "Non classé"


def upgrade() -> None:
    op.create_table(
        "ingested_files",
        sa.Column("filename", sa.String(512), primary_key=True),
        sa.Column("folder", sa.String(200), nullable=False, server_default=DEFAULT_FOLDER),
        sa.Column("subfolder", sa.String(200), nullable=False, server_default=DEFAULT_SUBFOLDER),
        sa.Column("updated_at", TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("idx_ingested_files_folder", "ingested_files", ["folder", "subfolder"])


def downgrade() -> None:
    op.drop_index("idx_ingested_files_folder", table_name="ingested_files")
    op.drop_table("ingested_files")
