import uuid
from datetime import datetime

from sqlalchemy import Index, String, Text
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
