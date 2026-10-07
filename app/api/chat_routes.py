"""Chatbot d'un cours : historique persistant en arbre de versions et questions limitées par jour.

Le cours est lu en base (jamais fourni par le client). Quota : `chat_daily_limit` questions par
cours et par jour UTC, comptées dans `course_chat_messages` (messages supprimés compris) ; vérifié
avant l'appel Gemini et consommé seulement quand l'échange est enregistré (un échec Gemini ne coûte
rien). Éditer une question crée une version sœur (même `parent_id`) et consomme aussi le quota ;
la suppression est logique (`deleted_at`) et emporte toute la descendance.
"""

import math
import uuid
from datetime import datetime, time, timedelta, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.api.routes import _gemini_http_errors, get_gemini_client
from app.api.schemas import ChatExchangeResponse, ChatHistoryResponse, ChatMessage, ChatQuota, ChatRequest
from app.core.config import Settings, get_settings
from app.core.rate_limit import SlidingWindowLimiter
from app.db.models import CourseChatMessage
from app.repositories import course_chat_repository, course_session_repository
from app.services.course_chat import answer_question, branch_history
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
        parent_id=message.parent_id,
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


def _parent_not_found() -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Message parent introuvable")


def _resolve_parent_id(body: ChatRequest, messages: list[CourseChatMessage]) -> UUID | None:
    """Réponse après laquelle s'insère la question (`messages` : messages actifs du cours).

    `parent_id` absent (anciens clients) : réponse la plus récente. Lève 404 si le parent fourni
    n'est pas une réponse active de ce cours.
    """
    if "parent_id" not in body.model_fields_set:
        return next((m.id for m in reversed(messages) if m.role == "assistant"), None)
    if body.parent_id is None:
        return None
    parent = next((m for m in messages if m.id == body.parent_id), None)
    if parent is None or parent.role != "assistant":
        raise _parent_not_found()
    return parent.id


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
    """Pose (ou réédite) une question au tuteur du cours, après la réponse `parent_id`.

    Le contexte envoyé au tuteur est la seule branche racine → `parent_id`. 404 si la session
    n'existe pas ou si `parent_id` n'est pas une réponse active de ce cours ; 422 si le message est
    vide ou trop long ; 429 au-delà de `chat_rate_limit_per_minute` (par IP) ou du quota journalier
    du cours (avec `Retry-After` jusqu'à minuit UTC) ; 502/503 (ou 429 quota Gemini) si Gemini
    échoue, sans consommer le quota.
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
        messages = await course_chat_repository.list_for_session(db, session_id)
        parent_id = _resolve_parent_id(body, messages)
        course = row.gemini_response

    history = branch_history(messages, parent_id)
    with _gemini_http_errors():
        reply = await answer_question(course, history, body.message, gemini_client, settings, body.section_id)

    user_id = uuid.uuid4()
    user = CourseChatMessage(
        id=user_id, session_id=session_id, parent_id=parent_id, role="user", content=body.message,
        status=reply.status, sources=[], created_at=received_at,
    )
    assistant = CourseChatMessage(
        id=uuid.uuid4(), session_id=session_id, parent_id=user_id, role="assistant", content=reply.content,
        status=reply.status, sources=reply.sources,
        created_at=max(_utc_now(), received_at + timedelta(microseconds=1)),
    )
    async with session_factory() as db:
        # Le parent a pu être supprimé pendant l'appel Gemini : verrou jusqu'au commit pour ne pas
        # rattacher l'échange à une branche supprimée.
        if parent_id is not None and await course_chat_repository.get_active_message(
            db, session_id, parent_id, for_update=True
        ) is None:
            raise _parent_not_found()
        user, assistant = await course_chat_repository.add_exchange(db, user, assistant)
    return ChatExchangeResponse(
        user_message=_to_api(user),
        assistant_message=_to_api(assistant),
        quota=_quota(used + 1, limit, resets_at),
    )


@router.delete("/courses/{session_id}/chat/messages/{message_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_course_chat_message(session_id: UUID, message_id: UUID, request: Request) -> Response:
    """Supprime (logiquement) une question, sa réponse et toute leur descendance.

    404 si la session n'existe pas ou si le message n'est pas une question active de ce cours.
    Ne rend pas de quota.
    """
    session_factory: async_sessionmaker = request.app.state.db_session_factory
    async with session_factory() as db:
        await _require_session(db, session_id)
        message = await course_chat_repository.get_active_message(db, session_id, message_id)
        if message is None or message.role != "user":
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Message introuvable")
        await course_chat_repository.soft_delete_branch(db, session_id, message_id, _utc_now())
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("/courses/{session_id}/chat", status_code=status.HTTP_204_NO_CONTENT)
async def delete_course_chat(session_id: UUID, request: Request) -> Response:
    """Supprime (logiquement) tout le chat du cours. 404 si la session n'existe pas. Ne rend pas de quota."""
    session_factory: async_sessionmaker = request.app.state.db_session_factory
    async with session_factory() as db:
        await _require_session(db, session_id)
        await course_chat_repository.soft_delete_all(db, session_id, _utc_now())
    return Response(status_code=status.HTTP_204_NO_CONTENT)
