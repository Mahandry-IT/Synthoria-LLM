import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import CourseSectionNote


async def get_for_session(session: AsyncSession, session_id: uuid.UUID) -> dict[str, str]:
    """Notes existantes d'une session, indexées par `section_id`."""
    result = await session.execute(select(CourseSectionNote).where(CourseSectionNote.session_id == session_id))
    return {r.section_id: r.note for r in result.scalars().all()}


async def upsert(session: AsyncSession, session_id: uuid.UUID, section_id: str, note: str) -> CourseSectionNote:
    """Crée ou remplace la note d'une section (chaîne vide = note effacée)."""
    row = await session.get(CourseSectionNote, (session_id, section_id))
    if row is None:
        row = CourseSectionNote(session_id=session_id, section_id=section_id, note=note)
        session.add(row)
    else:
        row.note = note
    await session.commit()
    await session.refresh(row)
    return row
