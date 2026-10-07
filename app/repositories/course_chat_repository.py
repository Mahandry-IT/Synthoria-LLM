import uuid
from datetime import datetime

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import CourseChatMessage


async def list_for_session(session: AsyncSession, session_id: uuid.UUID) -> list[CourseChatMessage]:
    """Messages non supprimés du chat d'un cours (toutes versions), en ordre chronologique croissant."""
    result = await session.execute(
        select(CourseChatMessage)
        .where(CourseChatMessage.session_id == session_id, CourseChatMessage.deleted_at.is_(None))
        .order_by(CourseChatMessage.created_at, CourseChatMessage.role.desc())
    )
    return list(result.scalars().all())


async def get_active_message(
    session: AsyncSession, session_id: uuid.UUID, message_id: uuid.UUID, *, for_update: bool = False
) -> CourseChatMessage | None:
    """Message non supprimé de ce cours, ou None. `for_update` verrouille la ligne jusqu'au commit."""
    statement = select(CourseChatMessage).where(
        CourseChatMessage.id == message_id,
        CourseChatMessage.session_id == session_id,
        CourseChatMessage.deleted_at.is_(None),
    )
    if for_update:
        statement = statement.with_for_update()
    result = await session.execute(statement)
    return result.scalar_one_or_none()


async def count_user_messages_since(session: AsyncSession, session_id: uuid.UUID, since: datetime) -> int:
    """Nombre de questions (`role=user`) posées sur ce cours depuis `since` (borne incluse).

    Les messages supprimés sont comptés : supprimer ne rend pas de quota.
    """
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
    """Enregistre la question et la réponse dans une même transaction (tout ou rien).

    La question est insérée d'abord : la réponse la référence (`parent_id`), relation que l'unité
    de travail ne connaît pas (pas de `relationship`).
    """
    session.add(user)
    await session.flush()
    session.add(assistant)
    await session.commit()
    await session.refresh(user)
    await session.refresh(assistant)
    return user, assistant


async def soft_delete_branch(
    session: AsyncSession, session_id: uuid.UUID, message_id: uuid.UUID, deleted_at: datetime
) -> int:
    """Supprime logiquement un message et toute sa descendance (réponse, suites, versions suivantes).

    Retourne le nombre de messages marqués.
    """
    branch = (
        select(CourseChatMessage.id)
        .where(CourseChatMessage.id == message_id, CourseChatMessage.session_id == session_id)
        .cte("branch", recursive=True)
    )
    branch = branch.union_all(
        select(CourseChatMessage.id).where(CourseChatMessage.parent_id == branch.c.id)
    )
    result = await session.execute(
        update(CourseChatMessage)
        .where(CourseChatMessage.id.in_(select(branch.c.id)), CourseChatMessage.deleted_at.is_(None))
        .values(deleted_at=deleted_at)
        .execution_options(synchronize_session=False)
    )
    await session.commit()
    return result.rowcount


async def soft_delete_all(session: AsyncSession, session_id: uuid.UUID, deleted_at: datetime) -> int:
    """Supprime logiquement tout le chat du cours. Retourne le nombre de messages marqués."""
    result = await session.execute(
        update(CourseChatMessage)
        .where(CourseChatMessage.session_id == session_id, CourseChatMessage.deleted_at.is_(None))
        .values(deleted_at=deleted_at)
        .execution_options(synchronize_session=False)
    )
    await session.commit()
    return result.rowcount
