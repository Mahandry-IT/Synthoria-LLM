"""Répétition espacée : cartes à réviser aujourd'hui et enregistrement du résultat.

Chaque carte présente une variante (0 = question « Vérifie » d'origine, ≥ 1 = variantes générées)
dans un mode déterministe (QCM ou réponse libre, voir `leitner.review_mode`).
"""

from datetime import datetime, timezone
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.api.routes import get_gemini_client
from app.core.config import Settings, get_settings
from app.core.rate_limit import SlidingWindowLimiter
from app.repositories import course_session_repository, flashcard_review_repository, flashcard_variant_repository
from app.repositories.flashcard_variant_repository import VariantIndex
from app.services import flashcard_variants
from app.services.gemini_client import GeminiClient
from app.services.leitner import card_back, flashcards_from_course, next_review, resolve_variant_no, review_mode

router = APIRouter(tags=["reviews"])


class DueCard(BaseModel):
    session_id: UUID
    course_title: str = ""
    card_id: str
    front: str
    back: str
    box: int = Field(0, description="Boîte de Leitner (0 = nouvelle ou ratée).")
    due_at: str | None = Field(None, description="Échéance ; None pour une carte jamais révisée.")
    variant_no: int = Field(0, description="Variante présentée (0 = question « Vérifie » d'origine).")
    mode: Literal["qcm", "text"] = Field(
        "text", description="QCM si (box + variant_no) est pair et que la carte a des choix, réponse libre sinon."
    )
    choices: list[str] = Field(default_factory=list)
    correct_indices: list[int] = Field(default_factory=list)
    explanation: str = ""


class DueCardList(BaseModel):
    cards: list[DueCard]
    total_due: int


class ReviewRequest(BaseModel):
    result: Literal["correct", "incorrect"]


class ReviewResponse(BaseModel):
    box: int
    due_at: str
    variant_no: int = Field(0, description="Variante prévue pour la prochaine échéance.")


def _limit_reviews(request: Request, settings: Settings = Depends(get_settings)) -> None:
    limiter = getattr(request.app.state, "review_rate_limiter", None)
    if limiter is None:
        limiter = request.app.state.review_rate_limiter = SlidingWindowLimiter()
    limiter.check(
        request.client.host if request.client else "unknown",
        settings.review_rate_limit_per_minute,
        "Trop de révisions, réessayez dans une minute",
    )


def _card_content(card: dict, session_id: UUID, stored_variant: int, variants: VariantIndex) -> tuple[int, dict]:
    """Variante affichable (repli sur la courante si la suivante n'est pas encore générée) et son contenu."""
    by_no = variants.get((session_id, card["card_id"]), {})
    variant_no = resolve_variant_no(stored_variant, set(by_no))
    if variant_no == 0:
        return 0, card
    variant = by_no[variant_no]
    return variant_no, {
        "front": variant.front,
        "back": card_back(variant.choices, variant.correct_indices, variant.explanation),
        "choices": variant.choices,
        "correct_indices": variant.correct_indices,
        "explanation": variant.explanation,
    }


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
        session_ids = [r.id for r in rows]
        reviews = await flashcard_review_repository.get_for_sessions(db, session_ids)
        variants = await flashcard_variant_repository.get_for_sessions(db, session_ids)

    due: list[tuple[datetime, DueCard]] = []
    for row in rows:
        title = (row.gemini_response.get("meta") or {}).get("title", "")
        for card in flashcards_from_course(row.gemini_response):
            review = reviews.get((row.id, card["card_id"]))
            if review is not None and review.due_at > now:
                continue
            box = review.box if review else 0
            variant_no, content = _card_content(card, row.id, review.variant_no if review else 0, variants)
            due.append(
                (
                    review.due_at if review else datetime.min.replace(tzinfo=timezone.utc),
                    DueCard(
                        session_id=row.id,
                        course_title=title,
                        card_id=card["card_id"],
                        front=content["front"],
                        back=content["back"],
                        box=box,
                        due_at=review.due_at.isoformat() if review else None,
                        variant_no=variant_no,
                        mode=review_mode(box, variant_no, bool(content["choices"])),
                        choices=content["choices"],
                        correct_indices=content["correct_indices"],
                        explanation=content["explanation"],
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
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),
    gemini_client: GeminiClient = Depends(get_gemini_client),
) -> ReviewResponse:
    """Enregistre le résultat d'une révision et planifie la suivante (Leitner). 404 si la carte n'existe pas.

    « correct » fait avancer la variante d'un cran pour la prochaine échéance ; si elle n'existe pas
    encore, un lot de variantes est généré en tâche de fond (1 appel Gemini pour le cours)."""
    session_factory: async_sessionmaker = request.app.state.db_session_factory
    async with session_factory() as db:
        row = await course_session_repository.get_by_id(db, session_id)
        if row is None or card_id not in {c["card_id"] for c in flashcards_from_course(row.gemini_response)}:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Carte introuvable")

        correct = body.result == "correct"
        current = await flashcard_review_repository.get_one(db, session_id, card_id)
        variants = await flashcard_variant_repository.get_for_sessions(db, [session_id])
        available = set(variants.get((session_id, card_id), {}))
        shown = resolve_variant_no(current.variant_no if current else 0, available)
        variant_no = shown + 1 if correct else shown
        box, due_at = next_review(
            current.box if current else None, correct, datetime.now(timezone.utc), settings.review_intervals_days
        )
        await flashcard_review_repository.upsert(
            db, session_id=session_id, card_id=card_id, box=box, due_at=due_at, last_result=body.result,
            variant_no=variant_no,
        )

    if variant_no not in available | {0}:
        _schedule_variants(request, background_tasks, session_id, session_factory, gemini_client, settings)
    return ReviewResponse(box=box, due_at=due_at.isoformat(), variant_no=variant_no)


def _schedule_variants(
    request: Request,
    background_tasks: BackgroundTasks,
    session_id: UUID,
    session_factory: async_sessionmaker,
    gemini_client: GeminiClient,
    settings: Settings,
) -> None:
    """Un seul lot de variantes en cours par cours (dans ce processus)."""
    in_flight = getattr(request.app.state, "flashcard_variants_in_flight", None)
    if in_flight is None:
        in_flight = request.app.state.flashcard_variants_in_flight = set()
    if session_id in in_flight:
        return
    in_flight.add(session_id)
    background_tasks.add_task(
        flashcard_variants.generate_variants, session_id, session_factory, gemini_client, settings, in_flight
    )
