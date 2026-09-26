import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import CourseVideoNote


async def get_for_session(session: AsyncSession, session_id: uuid.UUID) -> dict[str, str]:
    """Notes existantes d'une session, indexées par `video_id`."""
    result = await session.execute(select(CourseVideoNote).where(CourseVideoNote.session_id == session_id))
    return {r.video_id: r.note for r in result.scalars().all()}


async def upsert(session: AsyncSession, session_id: uuid.UUID, video_id: str, note: str) -> CourseVideoNote:
    """Crée ou remplace la note d'une vidéo (chaîne vide = note effacée)."""
    row = await session.get(CourseVideoNote, (session_id, video_id))
    if row is None:
        row = CourseVideoNote(session_id=session_id, video_id=video_id, note=note)
        session.add(row)
    else:
        row.note = note
    await session.commit()
    await session.refresh(row)
    return row
