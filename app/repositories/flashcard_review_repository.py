import uuid
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import FlashcardReview


async def get_for_sessions(session: AsyncSession, session_ids: list[uuid.UUID]) -> dict[tuple[uuid.UUID, str], FlashcardReview]:
    """Révisions existantes des sessions données, indexées par (session_id, card_id)."""
    if not session_ids:
        return {}
    result = await session.execute(select(FlashcardReview).where(FlashcardReview.session_id.in_(session_ids)))
    return {(r.session_id, r.card_id): r for r in result.scalars().all()}


async def get_one(session: AsyncSession, session_id: uuid.UUID, card_id: str) -> FlashcardReview | None:
    return await session.get(FlashcardReview, (session_id, card_id))


async def upsert(
    session: AsyncSession,
    *,
    session_id: uuid.UUID,
    card_id: str,
    box: int,
    due_at: datetime,
    last_result: str,
) -> FlashcardReview:
    """Crée ou met à jour la révision d'une carte."""
    review = await get_one(session, session_id, card_id)
    if review is None:
        review = FlashcardReview(session_id=session_id, card_id=card_id, box=box, due_at=due_at, last_result=last_result)
        session.add(review)
    else:
        review.box, review.due_at, review.last_result = box, due_at, last_result
    await session.commit()
    return review
