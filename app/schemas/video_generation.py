"""Pydantic schemas for video generation API responses."""

from __future__ import annotations

from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field


class VideoJobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class VideoGenerationJobCreate(BaseModel):
    """Réponse 202 Accepted lors de la création d'un job vidéo."""

    job_id: UUID
    status: VideoJobStatus = VideoJobStatus.PENDING
    course_session_id: UUID


class VideoGenerationJobResponse(BaseModel):
    """Détail d'un job de génération vidéo."""

    job_id: UUID
    status: VideoJobStatus
    model_used: str | None = None
    fallback_used: bool = False
    video_url: str | None = None
    error: str | None = None
    created_at: str
    updated_at: str
