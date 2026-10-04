"""Chatbot d'un cours : historique persistant et questions limitées par jour et par cours.

Le cours est lu en base (jamais fourni par le client). Quota : `chat_daily_limit` questions par
cours et par jour UTC, comptées dans `course_chat_messages` ; vérifié avant l'appel Gemini et
consommé seulement quand l'échange est enregistré (un échec Gemini ne coûte rien).
"""

import math
import uuid
from datetime import datetime, time, timedelta, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.api.routes import _gemini_http_errors, get_gemini_client
from app.api.schemas import ChatExchangeResponse, ChatHistoryResponse, ChatMessage, ChatQuota, ChatRequest
from app.core.config import Settings, get_settings
from app.core.rate_limit import SlidingWindowLimiter
from app.db.models import CourseChatMessage
from app.repositories import course_chat_repository, course_session_repository
from app.services.course_chat import answer_question
from app.services.gemini_client import GeminiClient

router = APIRouter(tags=["chat"])


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _day_bounds(now: datetime) -> tuple[datetime, datetime]:
    """Minuit UTC du jour courant et du lendemain (fenêtre du quota journalier)."""
    start = datetime.combine(now.astimezone(timezone.utc).date(), time.min, tzinfo=timezone.utc)
    return start, start + timedelta(days=1)


def _quota(used: int, limit: int, resets_at: datetime) -> ChatQuota:
    return ChatQuota(limit=limit, used=used, remaining=max(0, limit - used), resets_at=resets_at)


def _to_api(message: CourseChatMessage) -> ChatMessage:
    return ChatMessage(
        id=message.id,
        role=message.role,
        content=message.content,
        status=message.status,
        sources=message.sources or [],
        created_at=message.created_at,
    )


def _limit_chat(request: Request, settings: Settings = Depends(get_settings)) -> None:
    limiter = getattr(request.app.state, "chat_rate_limiter", None)
    if limiter is None:
        limiter = request.app.state.chat_rate_limiter = SlidingWindowLimiter()
    limiter.check(
        request.client.host if request.client else "unknown",
        settings.chat_rate_limit_per_minute,
        "Trop de messages envoyés au chat, réessayez dans une minute",
    )


async def _require_session(db, session_id: UUID):
    row = await course_session_repository.get_by_id(db, session_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session introuvable")
    return row


@router.get("/courses/{session_id}/chat", response_model=ChatHistoryResponse)
async def get_course_chat(
    session_id: UUID, request: Request, settings: Settings = Depends(get_settings)
) -> ChatHistoryResponse:
    """Historique du chat du cours (ordre chronologique) et quota du jour. 404 si la session n'existe pas."""
    day_start, resets_at = _day_bounds(_utc_now())
    session_factory: async_sessionmaker = request.app.state.db_session_factory
    async with session_factory() as db:
        await _require_session(db, session_id)
        messages = await course_chat_repository.list_for_session(db, session_id)
        used = await course_chat_repository.count_user_messages_since(db, session_id, day_start)
    return ChatHistoryResponse(
        messages=[_to_api(m) for m in messages], quota=_quota(used, settings.chat_daily_limit, resets_at)
    )


@router.post(
    "/courses/{session_id}/chat",
    response_model=ChatExchangeResponse,
    dependencies=[Depends(_limit_chat)],
)
async def post_course_chat(
    session_id: UUID,
    body: ChatRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
    gemini_client: GeminiClient = Depends(get_gemini_client),
) -> ChatExchangeResponse:
    """Pose une question au tuteur du cours.

    404 si la session n'existe pas ; 422 si le message est vide ou trop long ; 429 au-delà de
    `chat_rate_limit_per_minute` (par IP) ou du quota journalier du cours (avec `Retry-After`
    jusqu'à minuit UTC) ; 502/503 (ou 429 quota Gemini) si Gemini échoue, sans consommer le quota.
    """
    received_at = _utc_now()
    day_start, resets_at = _day_bounds(received_at)
    limit = settings.chat_daily_limit
    session_factory: async_sessionmaker = request.app.state.db_session_factory

    # Lecture dans une session courte : la connexion n'est pas retenue pendant l'appel Gemini.
    async with session_factory() as db:
        row = await _require_session(db, session_id)
        used = await course_chat_repository.count_user_messages_since(db, session_id, day_start)
        if used >= limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"Limite de {limit} messages par jour atteinte pour ce cours. "
                    "Vous pourrez poser de nouvelles questions après minuit (UTC)."
                ),
                headers={"Retry-After": str(max(1, math.ceil((resets_at - received_at).total_seconds())))},
            )
        history = await course_chat_repository.list_for_session(db, session_id)
        course = row.gemini_response

    with _gemini_http_errors():
        reply = await answer_question(course, history, body.message, gemini_client, settings)

    user = CourseChatMessage(
        id=uuid.uuid4(), session_id=session_id, role="user", content=body.message, status=reply.status, sources=[],
        created_at=received_at,
    )
    assistant = CourseChatMessage(
        id=uuid.uuid4(), session_id=session_id, role="assistant", content=reply.content, status=reply.status,
        sources=reply.sources, created_at=max(_utc_now(), received_at + timedelta(microseconds=1)),
    )
    async with session_factory() as db:
        user, assistant = await course_chat_repository.add_exchange(db, user, assistant)
    return ChatExchangeResponse(
        user_message=_to_api(user),
        assistant_message=_to_api(assistant),
        quota=_quota(used + 1, limit, resets_at),
    )
