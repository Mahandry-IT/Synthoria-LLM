"""create course_plans table

Revision ID: 003
Revises: 001
Create Date: 2026-09-18
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

# revision identifiers
revision = '003'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'course_plans',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column('question', sa.Text(), nullable=False),
        sa.Column('mode', sa.String(20), nullable=False),
        sa.Column('filenames', JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column('top_k', sa.Integer(), nullable=False),
        sa.Column('full_document', sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column('retrieval_context', JSONB(), nullable=False),
        sa.Column('plan', JSONB(), nullable=False),
        sa.Column('status', sa.String(20), nullable=False, server_default='pending'),
        sa.Column('expires_at', sa.TIMESTAMP(timezone=True), nullable=False),
    )
    op.create_index('idx_course_plans_expires_at', 'course_plans', ['expires_at'])


def downgrade() -> None:
    op.drop_index('idx_course_plans_expires_at', table_name='course_plans')
    op.drop_table('course_plans')
