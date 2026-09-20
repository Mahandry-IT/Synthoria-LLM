"""add podcast_jobs table

Revision ID: 004
Revises: 002_add_video_generation_jobs
Create Date: 2026-09-20
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

# revision identifiers
revision = "004"
down_revision = "002_add_video_generation_jobs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "podcast_jobs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column(
            "course_session_id",
            UUID(as_uuid=True),
            sa.ForeignKey("course_sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("stage", sa.String(20), nullable=True),
        sa.Column("progress", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("params", JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("params_hash", sa.String(64), nullable=False),
        sa.Column("script", JSONB(), nullable=True),
        sa.Column("audio_path", sa.Text(), nullable=True),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column("tts_engine", sa.String(50), nullable=True),
        sa.Column("voices", JSONB(), nullable=True),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("locked_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("idx_podcast_jobs_status_created", "podcast_jobs", ["status", "created_at"])
    op.create_index("idx_podcast_jobs_session_params", "podcast_jobs", ["course_session_id", "params_hash"])


def downgrade() -> None:
    op.drop_index("idx_podcast_jobs_session_params", table_name="podcast_jobs")
    op.drop_index("idx_podcast_jobs_status_created", table_name="podcast_jobs")
    op.drop_table("podcast_jobs")
