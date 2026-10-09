"""add quiz_questions (question bank) and quiz_attempts (server-graded attempts)

Revision ID: 019
Revises: 018
Create Date: 2026-10-09

Tables aussi créées au démarrage de l'API par `create_all` (bases sans `alembic_version`).
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, UUID

# revision identifiers
revision = "019"
down_revision = "018"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "quiz_questions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "session_id", UUID(as_uuid=True),
            sa.ForeignKey("course_sessions.id", ondelete="CASCADE"), nullable=False,
        ),
        sa.Column("payload", JSONB, nullable=False),
        sa.Column("batch", sa.Integer, nullable=False, server_default="0"),
        sa.Column("times_served", sa.Integer, nullable=False, server_default="0"),
        sa.Column("last_served_at", TIMESTAMP(timezone=True), nullable=True),
        sa.Column("created_at", TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("idx_quiz_questions_session_served", "quiz_questions", ["session_id", "times_served"])

    op.create_table(
        "quiz_attempts",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "session_id", UUID(as_uuid=True),
            sa.ForeignKey("course_sessions.id", ondelete="CASCADE"), nullable=False,
        ),
        sa.Column("question_ids", JSONB, nullable=False, server_default="[]"),
        sa.Column("answers", JSONB, nullable=True),
        sa.Column("score", sa.Numeric(5, 2), nullable=True),
        sa.Column("max_score", sa.Numeric(5, 2), nullable=False, server_default="20"),
        sa.Column("status", sa.String(16), nullable=False, server_default="in_progress"),
        sa.Column("abort_reason", sa.String(200), nullable=True),
        sa.Column("started_at", TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("finished_at", TIMESTAMP(timezone=True), nullable=True),
    )
    op.create_index("idx_quiz_attempts_session_started", "quiz_attempts", ["session_id", "started_at"])


def downgrade() -> None:
    op.drop_index("idx_quiz_attempts_session_started", table_name="quiz_attempts")
    op.drop_table("quiz_attempts")
    op.drop_index("idx_quiz_questions_session_served", table_name="quiz_questions")
    op.drop_table("quiz_questions")
