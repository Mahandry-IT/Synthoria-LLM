import uuid
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import CourseChatMessage


async def list_for_session(session: AsyncSession, session_id: uuid.UUID) -> list[CourseChatMessage]:
    """Historique complet du chat d'un cours, en ordre chronologique croissant."""
    result = await session.execute(
        select(CourseChatMessage)
        .where(CourseChatMessage.session_id == session_id)
        .order_by(CourseChatMessage.created_at, CourseChatMessage.role.desc())
    )
    return list(result.scalars().all())


async def count_user_messages_since(session: AsyncSession, session_id: uuid.UUID, since: datetime) -> int:
    """Nombre de questions (`role=user`) posées sur ce cours depuis `since` (borne incluse)."""
    result = await session.execute(
        select(func.count())
        .select_from(CourseChatMessage)
        .where(
            CourseChatMessage.session_id == session_id,
            CourseChatMessage.role == "user",
            CourseChatMessage.created_at >= since,
        )
    )
    return int(result.scalar_one())


async def add_exchange(
    session: AsyncSession, user: CourseChatMessage, assistant: CourseChatMessage
) -> tuple[CourseChatMessage, CourseChatMessage]:
    """Enregistre la question et la réponse dans une même transaction (tout ou rien)."""
    session.add_all([user, assistant])
    await session.commit()
    await session.refresh(user)
    await session.refresh(assistant)
    return user, assistant
