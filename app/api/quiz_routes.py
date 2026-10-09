"""Tentatives de quiz notées côté serveur, tirées d'une banque de questions renouvelée par cours."""

import uuid
from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.api.routes import get_gemini_client
from app.api.schemas import QuizQuestion
from app.core.config import Settings, get_settings
from app.core.rate_limit import SlidingWindowLimiter
from app.services import quiz_bank
from app.services.gemini_client import GeminiClient

router = APIRouter(tags=["quiz"])

_MAX_QUESTIONS = 100
_MAX_CHOICES = 10
_MAX_ABORT_REASON = 200


class QuizAttemptStartResponse(BaseModel):
    attempt_id: uuid.UUID
    questions: list[QuizQuestion] = Field(
        description="Questions tirées de la banque ; points recalculés (total 20) ; bonnes réponses et explications vides."
    )


class QuizAttemptSubmitRequest(BaseModel):
    answers: list[Annotated[list[Annotated[int, Field(ge=0)]], Field(max_length=_MAX_CHOICES)]] = Field(
        max_length=_MAX_QUESTIONS,
        description="Indices 0-based choisis, une liste par question dans l'ordre de la tentative ([] = sans réponse).",
    )
    aborted: bool = False
    abort_reason: str | None = Field(None, max_length=_MAX_ABORT_REASON)


class QuizQuestionResult(BaseModel):
    question: str
    answer: list[int]
    correct_option_indices: list[int]
    is_correct: bool
    points: float
    points_earned: float
    explanation: str = ""
    explanation_per_choice: list[str] = Field(default_factory=list)


class QuizAttemptSubmitResponse(BaseModel):
    score: float | None = Field(description="Note /20 ; null pour une tentative interrompue.")
    max_score: float = quiz_bank.QUIZ_MAX_SCORE
    status: Literal["completed", "aborted"]
    results: list[QuizQuestionResult] = Field(description="Correction par question ; vide si interrompue.")


class QuizAttemptSummary(BaseModel):
    attempt_id: uuid.UUID
    started_at: datetime
    finished_at: datetime | None
    score: float | None
    max_score: float
    status: Literal["in_progress", "completed", "aborted"]
    abort_reason: str | None = None


class QuizAttemptList(BaseModel):
    attempts: list[QuizAttemptSummary]


def _limiter(request: Request, name: str) -> SlidingWindowLimiter:
    limiter = getattr(request.app.state, name, None)
    if limiter is None:
        limiter = SlidingWindowLimiter()
        setattr(request.app.state, name, limiter)
    return limiter


def _client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _limit_start(request: Request, settings: Settings = Depends(get_settings)) -> None:
    _limiter(request, "quiz_start_rate_limiter").check(
        _client_key(request), settings.quiz_attempt_rate_limit_per_minute,
        "Trop de tentatives démarrées, réessayez dans une minute",
    )


def _limit_submit(request: Request, settings: Settings = Depends(get_settings)) -> None:
    _limiter(request, "quiz_submit_rate_limiter").check(
        _client_key(request), settings.quiz_submit_rate_limit_per_minute,
        "Trop de soumissions, réessayez dans une minute",
    )


def _refills_in_flight(request: Request) -> set[uuid.UUID]:
    in_flight = getattr(request.app.state, "quiz_refills_in_flight", None)
    if in_flight is None:
        in_flight = request.app.state.quiz_refills_in_flight = set()
    return in_flight


@router.post(
    "/courses/{session_id}/quiz/attempts",
    response_model=QuizAttemptStartResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(_limit_start)],
)
async def start_quiz_attempt(
    session_id: uuid.UUID,
    request: Request,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),
    gemini_client: GeminiClient = Depends(get_gemini_client),
) -> QuizAttemptStartResponse:
    """Démarre une tentative : nouvelle série tirée de la banque (amorcée au premier appel).
    Déclenche en tâche de fond la recharge de la banque si elle s'épuise, sans l'attendre."""
    session_factory: async_sessionmaker = request.app.state.db_session_factory
    async with session_factory() as db:
        try:
            started = await quiz_bank.start_attempt(db, session_id)
        except quiz_bank.QuizSessionNotFound:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session introuvable")
        except quiz_bank.QuizUnavailable:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ce cours n'a pas de quiz")

    in_flight = _refills_in_flight(request)
    if started.refill_needed and session_id not in in_flight:
        in_flight.add(session_id)
        background_tasks.add_task(quiz_bank.refill_bank, session_id, session_factory, gemini_client, settings, in_flight)

    return QuizAttemptStartResponse(attempt_id=started.attempt_id, questions=started.questions)


@router.post(
    "/courses/{session_id}/quiz/attempts/{attempt_id}/submit",
    response_model=QuizAttemptSubmitResponse,
    dependencies=[Depends(_limit_submit)],
)
async def submit_quiz_attempt(
    session_id: uuid.UUID,
    attempt_id: uuid.UUID,
    body: QuizAttemptSubmitRequest,
    request: Request,
) -> QuizAttemptSubmitResponse:
    """Corrige la tentative côté serveur (une seule fois) ; `aborted` la clôt sans note."""
    session_factory: async_sessionmaker = request.app.state.db_session_factory
    async with session_factory() as db:
        try:
            submitted = await quiz_bank.submit_attempt(
                db, session_id, attempt_id, body.answers,
                aborted=body.aborted, abort_reason=body.abort_reason if body.aborted else None,
            )
        except quiz_bank.QuizAttemptNotFound:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tentative introuvable")
        except quiz_bank.QuizAttemptAlreadySubmitted:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Tentative déjà soumise")
        except quiz_bank.InvalidQuizAnswers as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Réponses invalides : {exc}")

    return QuizAttemptSubmitResponse(
        score=submitted.score, max_score=submitted.max_score, status=submitted.status, results=submitted.results
    )


@router.get("/courses/{session_id}/quiz/attempts", response_model=QuizAttemptList)
async def list_quiz_attempts(
    session_id: uuid.UUID,
    request: Request,
    limit: int = Query(50, ge=1, le=200),
) -> QuizAttemptList:
    """Historique des tentatives du cours, les plus récentes d'abord."""
    session_factory: async_sessionmaker = request.app.state.db_session_factory
    async with session_factory() as db:
        try:
            attempts = await quiz_bank.list_attempts(db, session_id, limit)
        except quiz_bank.QuizSessionNotFound:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session introuvable")
    return QuizAttemptList(
        attempts=[
            QuizAttemptSummary(
                attempt_id=a.id, started_at=a.started_at, finished_at=a.finished_at,
                score=a.score, max_score=a.max_score, status=a.status, abort_reason=a.abort_reason,
            )
            for a in attempts
        ]
    )
