import uuid
from datetime import datetime

from sqlalchemy import Boolean, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class CourseSession(Base):
    """Session de génération de cours, persistée en PostgreSQL."""

    __tablename__ = "course_sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=datetime.utcnow, nullable=False
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    filenames: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    mode: Mapped[str] = mapped_column(String(20), nullable=False)
    gemini_response: Mapped[dict] = mapped_column(JSONB, nullable=False)

    __table_args__ = (
        Index("idx_course_sessions_created_at", created_at.desc()),
    )


class CoursePlan(Base):
    """Plan de cours proposé, en attente de validation par l'utilisateur.

    Le contexte de récupération (RAG / recherche web) est figé au moment du
    plan : le cours validé est généré à partir du même contexte, pas d'un
    nouveau retrieval potentiellement différent.
    """

    __tablename__ = "course_plans"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=datetime.utcnow, nullable=False
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    mode: Mapped[str] = mapped_column(String(20), nullable=False)
    filenames: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    top_k: Mapped[int] = mapped_column(Integer, nullable=False)
    full_document: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    retrieval_context: Mapped[dict] = mapped_column(JSONB, nullable=False)
    plan: Mapped[dict] = mapped_column(JSONB, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    expires_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)

    __table_args__ = (
        Index("idx_course_plans_expires_at", expires_at),
    )


class PodcastJob(Base):
    """Job asynchrone de génération de podcast à partir d'un cours persisté.

    Sert aussi de file d'attente : un worker réclame les jobs `pending` (ou
    périmés) via `SELECT ... FOR UPDATE SKIP LOCKED`. `script` est le
    checkpoint de l'étape de scriptage (reprise sans rappeler Gemini).
    """

    __tablename__ = "podcast_jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    course_session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("course_sessions.id", ondelete="CASCADE"), nullable=False
    )
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    stage: Mapped[str | None] = mapped_column(String(20), nullable=True)
    progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    params: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    params_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    script: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    audio_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    tts_engine: Mapped[str | None] = mapped_column(String(50), nullable=True)
    voices: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    locked_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    __table_args__ = (
        Index("idx_podcast_jobs_status_created", "status", "created_at"),
        Index("idx_podcast_jobs_session_params", "course_session_id", "params_hash"),
    )


class FlashcardReview(Base):
    """État de répétition espacée (Leitner) d'une flashcard d'un cours persisté."""

    __tablename__ = "flashcard_reviews"

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("course_sessions.id", ondelete="CASCADE"), primary_key=True
    )
    card_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    box: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    due_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    last_result: Mapped[str | None] = mapped_column(String(10), nullable=True)

    __table_args__ = (Index("idx_flashcard_reviews_due_at", due_at),)
