"""Routes des exercices corrigés : lancement (202), état et liste (sans énoncé), détail (sans solution), correction.

`expected_solution` ne sort jamais d'ici ; la réponse de l'apprenant n'est jamais enregistrée.
"""

import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.api.routes import _gemini_http_errors, get_gemini_client
from app.api.schemas import ApiContentBlock
from app.core.config import Settings, get_settings
from app.core.errors import ApiError
from app.core.rate_limit import SlidingWindowLimiter
from app.repositories import course_session_repository, exercise_job_repository, exercise_repository, quiz_repository
from app.services import quiz_bank
from app.services.exercise_evaluator import evaluate_exercise_answer
from app.services.exercise_generator import course_is_ready
from app.services.exercise_jobs import CourseSessionNotFoundError, enqueue_exercise_job
from app.services.gemini_client import GeminiClient

router = APIRouter(tags=["exercises"])

ExerciseStatus = Literal["absent", "pending", "generating", "done", "failed"]
# Borne absolue du schéma ; la limite configurable (`exercise_answer_max_length`) est vérifiée en plus.
_ANSWER_HARD_MAX = 20_000


class ExerciseGenerateResponse(BaseModel):
    status: Literal["pending", "generating", "done"]


class ExerciseSummary(BaseModel):
    """Entrée de la liste : jamais l'énoncé, les indices ni la solution (RG-8)."""

    id: uuid.UUID
    section_id: str | None
    position: int
    title: str
    covers: str
    duration_minutes: int


class ExerciseListResponse(BaseModel):
    status: ExerciseStatus
    error_message: str | None = None
    exercises: list[ExerciseSummary] = Field(default_factory=list)


class ExerciseDetail(BaseModel):
    """Énoncé et indices, sans solution : appelé au clic sur « Commencer »."""

    id: uuid.UUID
    section_id: str | None
    position: int
    title: str
    covers: str
    statement: list[ApiContentBlock]
    hints: list[str]
    duration_minutes: int


class ExerciseAnswerRequest(BaseModel):
    answer: str = Field(..., min_length=1, max_length=_ANSWER_HARD_MAX, description="Réponse de l'apprenant.")

    @field_validator("answer")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("answer ne peut pas être vide")
        return v


class ExerciseAnswerResponse(BaseModel):
    """Correction : aucun score (RG-1)."""

    feedback: str
    correct_points: list[str] = Field(default_factory=list)
    missing_points: list[str] = Field(default_factory=list)
    expected_approach: str


def _limiter(request: Request, name: str) -> SlidingWindowLimiter:
    limiter = getattr(request.app.state, name, None)
    if limiter is None:
        limiter = SlidingWindowLimiter()
        setattr(request.app.state, name, limiter)
    return limiter


def _client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _limit_generate(request: Request, settings: Settings = Depends(get_settings)) -> None:
    _limiter(request, "exercise_generate_rate_limiter").check(
        _client_key(request), settings.exercise_generate_rate_limit_per_minute,
        "Trop de demandes d'exercices, réessayez dans une minute",
    )


def _limit_answer(request: Request, settings: Settings = Depends(get_settings)) -> None:
    _limiter(request, "exercise_answer_rate_limiter").check(
        _client_key(request), settings.exercise_answer_rate_limit_per_minute,
        "Trop de corrections demandées, réessayez dans une minute",
    )


def _session_factory(request: Request) -> async_sessionmaker:
    return request.app.state.db_session_factory


def _not_found(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)


async def _require_session(db, session_id: uuid.UUID) -> None:
    if not await quiz_repository.session_exists(db, session_id):
        raise _not_found("Session introuvable")


@router.post(
    "/courses/{session_id}/exercises/generate",
    response_model=ExerciseGenerateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(_limit_generate)],
)
async def generate_exercises(
    session_id: uuid.UUID, request: Request, settings: Settings = Depends(get_settings)
) -> ExerciseGenerateResponse:
    """Met en file la génération du lot d'exercices du cours (202), sur demande seulement (RG-5).

    Idempotent : un seul lot par cours, relancé seulement s'il a échoué. `404` session inconnue,
    `409` si le cours ou son quiz n'est pas terminé (RG-4), `429` au-delà de la limite.
    """
    factory = _session_factory(request)
    async with factory() as db:
        course = await course_session_repository.get_by_id(db, session_id)
    if course is None:
        raise _not_found("Session introuvable")
    if not course_is_ready(course.gemini_response):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Le cours n'est pas terminé")
    if not quiz_bank.original_quiz(course.gemini_response):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Le quiz du cours n'est pas terminé")
    try:
        job = await enqueue_exercise_job(factory, session_id, course.gemini_response, settings)
    except CourseSessionNotFoundError as exc:
        raise _not_found("Session introuvable") from exc
    return ExerciseGenerateResponse(status=job.status)


@router.get("/courses/{session_id}/exercises", response_model=ExerciseListResponse)
async def list_exercises(session_id: uuid.UUID, request: Request) -> ExerciseListResponse:
    """État de la génération et liste des exercices (titre, section, ce qu'ils couvrent), sans énoncé."""
    async with _session_factory(request)() as db:
        await _require_session(db, session_id)
        job = await exercise_job_repository.get_by_session(db, session_id)
        if job is None:
            return ExerciseListResponse(status="absent")
        exercises = await exercise_repository.list_by_session(db, session_id) if job.status == "done" else []
    return ExerciseListResponse(
        status=job.status,
        error_message=job.error_message if job.status == "failed" else None,
        exercises=[
            ExerciseSummary(
                id=e.id, section_id=e.section_id, position=e.position, title=e.title, covers=e.covers,
                duration_minutes=e.duration_minutes,
            )
            for e in exercises
        ],
    )


@router.get("/courses/{session_id}/exercises/{exercise_id}", response_model=ExerciseDetail)
async def get_exercise(session_id: uuid.UUID, exercise_id: uuid.UUID, request: Request) -> ExerciseDetail:
    """Énoncé, indices et durée d'un exercice, sans solution. `404` si l'exercice n'appartient pas à la session."""
    async with _session_factory(request)() as db:
        exercise = await exercise_repository.get_for_session(db, session_id, exercise_id)
    if exercise is None:
        raise _not_found("Exercice introuvable")
    return ExerciseDetail(
        id=exercise.id, section_id=exercise.section_id, position=exercise.position, title=exercise.title,
        covers=exercise.covers, statement=exercise.statement, hints=exercise.hints,
        duration_minutes=exercise.duration_minutes,
    )


@router.post(
    "/courses/{session_id}/exercises/{exercise_id}/answer",
    response_model=ExerciseAnswerResponse,
    dependencies=[Depends(_limit_answer)],
)
async def answer_exercise(
    session_id: uuid.UUID,
    exercise_id: uuid.UUID,
    body: ExerciseAnswerRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
    gemini_client: GeminiClient = Depends(get_gemini_client),
) -> ExerciseAnswerResponse:
    """Corrige la réponse de l'apprenant (ni score, ni enregistrement de la réponse).

    L'énoncé et la solution sont lus en base. `404` exercice inconnu pour cette session, `422` réponse vide
    ou trop longue, `429` limite ou quota Gemini, `502`/`503` échec Gemini : le client garde sa réponse et
    peut réessayer.
    """
    if len(body.answer) > settings.exercise_answer_max_length:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Réponse trop longue")
    async with _session_factory(request)() as db:
        exercise = await exercise_repository.get_for_session(db, session_id, exercise_id)
    if exercise is None:
        raise _not_found("Exercice introuvable")
    try:
        with _gemini_http_errors():
            evaluation = await evaluate_exercise_answer(exercise, body.answer, gemini_client)
    except ApiError as exc:
        raise ApiError(
            exc.status_code, f"{exc.detail} Réessayez dans un instant : votre réponse est conservée.",
            exc.error_code, headers=exc.headers, debug=exc.debug,
        ) from exc
    return ExerciseAnswerResponse(
        feedback=evaluation.feedback, correct_points=evaluation.correct_points,
        missing_points=evaluation.missing_points, expected_approach=evaluation.expected_approach,
    )
