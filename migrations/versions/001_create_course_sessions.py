"""create course_sessions table

Revision ID: 001
Revises:
Create Date: 2026-08-23
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

# revision identifiers
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    op.create_table(
        'course_sessions',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column('question', sa.Text(), nullable=False),
        sa.Column('filenames', JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column('mode', sa.String(20), nullable=False),
        sa.Column('gemini_response', JSONB(), nullable=False),
    )
    op.create_index('idx_course_sessions_created_at', 'course_sessions', [sa.text('created_at DESC')])


def downgrade() -> None:
    op.drop_index('idx_course_sessions_created_at', table_name='course_sessions')
    op.drop_table('course_sessions')
