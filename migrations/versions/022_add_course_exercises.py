"""add exercise_jobs and course_exercises tables

Revision ID: 022
Revises: 021
Create Date: 2026-11-13

Exercices corrigés (lot 3) : file de génération (une ligne par cours) et exercices produits. Les bases
créées par `create_all` reçoivent ces tables au démarrage de l'API.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, UUID

# revision identifiers
revision = "022"
down_revision = "021"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "exercise_jobs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column(
            "session_id", UUID(as_uuid=True),
            sa.ForeignKey("course_sessions.id", ondelete="CASCADE"), nullable=False, unique=True,
        ),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("params_hash", sa.String(64), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("locked_at", TIMESTAMP(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("idx_exercise_jobs_status_created", "exercise_jobs", ["status", "created_at"])

    op.create_table(
        "course_exercises",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column(
            "session_id", UUID(as_uuid=True),
            sa.ForeignKey("course_sessions.id", ondelete="CASCADE"), nullable=False,
        ),
        sa.Column("section_id", sa.String(64), nullable=True),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(300), nullable=False),
        sa.Column("covers", sa.Text(), nullable=False),
        sa.Column("statement", JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("hints", JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("duration_minutes", sa.Integer(), nullable=False, server_default="10"),
        sa.Column("expected_solution", sa.Text(), nullable=False),
        sa.Column("created_at", TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("session_id", "position", name="uq_course_exercises_session_position"),
    )


def downgrade() -> None:
    op.drop_table("course_exercises")
    op.drop_index("idx_exercise_jobs_status_created", table_name="exercise_jobs")
    op.drop_table("exercise_jobs")
