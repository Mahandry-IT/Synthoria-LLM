"""Répétition espacée : cartes à réviser aujourd'hui et enregistrement du résultat."""

from datetime import datetime, timezone
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.core.config import Settings, get_settings
from app.core.rate_limit import SlidingWindowLimiter
from app.repositories import course_session_repository, flashcard_review_repository
from app.services.leitner import flashcards_from_course, next_review

router = APIRouter(tags=["reviews"])


class DueCard(BaseModel):
    session_id: UUID
    course_title: str = ""
    card_id: str
    front: str
    back: str
    box: int = Field(0, description="Boîte de Leitner (0 = nouvelle ou ratée).")
    due_at: str | None = Field(None, description="Échéance ; None pour une carte jamais révisée.")


class DueCardList(BaseModel):
    cards: list[DueCard]
    total_due: int


class ReviewRequest(BaseModel):
    result: Literal["correct", "incorrect"]


class ReviewResponse(BaseModel):
    box: int
    due_at: str


def _limit_reviews(request: Request, settings: Settings = Depends(get_settings)) -> None:
    limiter = getattr(request.app.state, "review_rate_limiter", None)
    if limiter is None:
        limiter = request.app.state.review_rate_limiter = SlidingWindowLimiter()
    limiter.check(
        request.client.host if request.client else "unknown",
        settings.review_rate_limit_per_minute,
        "Trop de révisions, réessayez dans une minute",
    )


@router.get("/reviews/due", response_model=DueCardList)
async def list_due_cards(
    request: Request,
    limit: int = Query(20, ge=1, le=100),
    settings: Settings = Depends(get_settings),
) -> DueCardList:
    """Cartes à réviser (jamais révisées ou échues) parmi les cours récents, les plus en retard d'abord."""
    now = datetime.now(timezone.utc)
    session_factory: async_sessionmaker = request.app.state.db_session_factory
    async with session_factory() as db:
        rows, _ = await course_session_repository.list_paginated(db, page=1, limit=settings.review_sessions_scan_limit)
        reviews = await flashcard_review_repository.get_for_sessions(db, [r.id for r in rows])

    due: list[tuple[datetime, DueCard]] = []
    for row in rows:
        title = (row.gemini_response.get("meta") or {}).get("title", "")
        for card in flashcards_from_course(row.gemini_response):
            review = reviews.get((row.id, card["card_id"]))
            if review is not None and review.due_at > now:
                continue
            due.append(
                (
                    review.due_at if review else datetime.min.replace(tzinfo=timezone.utc),
                    DueCard(
                        session_id=row.id,
                        course_title=title,
                        card_id=card["card_id"],
                        front=card["front"],
                        back=card["back"],
                        box=review.box if review else 0,
                        due_at=review.due_at.isoformat() if review else None,
                    ),
                )
            )
    due.sort(key=lambda item: item[0])
    return DueCardList(cards=[card for _, card in due[:limit]], total_due=len(due))


@router.post(
    "/reviews/{session_id}/{card_id}",
    response_model=ReviewResponse,
    dependencies=[Depends(_limit_reviews)],
)
async def record_review(
    session_id: UUID,
    card_id: str,
    body: ReviewRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> ReviewResponse:
    """Enregistre le résultat d'une révision et planifie la suivante (Leitner). 404 si la carte n'existe pas."""
    session_factory: async_sessionmaker = request.app.state.db_session_factory
    async with session_factory() as db:
        row = await course_session_repository.get_by_id(db, session_id)
        if row is None or card_id not in {c["card_id"] for c in flashcards_from_course(row.gemini_response)}:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Carte introuvable")

        current = await flashcard_review_repository.get_one(db, session_id, card_id)
        box, due_at = next_review(
            current.box if current else None,
            body.result == "correct",
            datetime.now(timezone.utc),
            settings.review_intervals_days,
        )
        await flashcard_review_repository.upsert(
            db, session_id=session_id, card_id=card_id, box=box, due_at=due_at, last_result=body.result
        )
    return ReviewResponse(box=box, due_at=due_at.isoformat())
