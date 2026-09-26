import uuid
from typing import Any

from sqlalchemy import asc, desc, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import CourseGenerationResponse
from app.db.models import DEFAULT_COURSE_FOLDER, DEFAULT_COURSE_SUBFOLDER, CourseSession


async def save(
    session: AsyncSession,
    *,
    question: str,
    filenames: list[str],
    mode: str,
    response: CourseGenerationResponse,
) -> CourseSession:
    """Persiste une session de génération de cours."""
    course_session = CourseSession(
        question=question,
        filenames=filenames,
        mode=mode,
        gemini_response=response.model_dump(exclude={"session_id", "podcast_job_id"}),
    )
    session.add(course_session)
    await session.commit()
    await session.refresh(course_session)
    return course_session


async def list_paginated(
    session: AsyncSession,
    *,
    page: int,
    limit: int,
    folder: str | None = None,
    subfolder: str | None = None,
) -> tuple[list[CourseSession], int]:
    """Liste paginée des sessions (plus récentes en premier), filtrable par dossier/sous-dossier."""
    offset = (page - 1) * limit
    conditions = []
    if folder is not None:
        conditions.append(CourseSession.folder == folder)
    if subfolder is not None:
        conditions.append(CourseSession.subfolder == subfolder)

    # Total
    count_result = await session.execute(select(func.count(CourseSession.id)).where(*conditions))
    total = count_result.scalar_one()

    # Data
    stmt = (
        select(CourseSession)
        .where(*conditions)
        .order_by(CourseSession.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    result = await session.execute(stmt)
    rows = list(result.scalars().all())

    return rows, total


async def list_folders(session: AsyncSession) -> list[tuple[str, str, int]]:
    """Dossiers/sous-dossiers réellement utilisés (au moins un cours), avec leur nombre de cours.

    Un dossier n'est qu'un attribut des cours (aucune entité dossier séparée) : il n'existe qu'en
    tant que valeur portée par au moins un cours, et sort de cette liste dès qu'aucun cours n'y est
    plus rangé.
    """
    stmt = (
        select(CourseSession.folder, CourseSession.subfolder, func.count(CourseSession.id))
        .group_by(CourseSession.folder, CourseSession.subfolder)
    )
    result = await session.execute(stmt)
    return list(result.all())


async def resolve_folder_casing(
    session: AsyncSession, folder: str, subfolder: str | None = None
) -> tuple[str, str | None]:
    """Résout la casse canonique d'un dossier (et d'un sous-dossier) déjà utilisés en base, par
    comparaison insensible à la casse — pour que « sfi » tapé quand « Sfi » existe déjà rejoigne
    « Sfi » au lieu de créer un dossier distinct qui ne diffère que par la casse.

    Casse canonique = celle utilisée par le plus de cours (égalité : la plus ancienne). Retourne la
    valeur fournie telle quelle si aucune variante n'existe déjà (nouveau dossier/sous-dossier).
    """
    folder_stmt = (
        select(CourseSession.folder, func.count().label("n"), func.min(CourseSession.created_at).label("first"))
        .where(func.lower(CourseSession.folder) == folder.lower())
        .group_by(CourseSession.folder)
        .order_by(desc("n"), asc("first"))
        .limit(1)
    )
    folder_row = (await session.execute(folder_stmt)).first()
    canonical_folder = folder_row.folder if folder_row else folder

    if not subfolder:
        return canonical_folder, subfolder

    subfolder_stmt = (
        select(CourseSession.subfolder, func.count().label("n"), func.min(CourseSession.created_at).label("first"))
        .where(
            func.lower(CourseSession.folder) == canonical_folder.lower(),
            func.lower(CourseSession.subfolder) == subfolder.lower(),
        )
        .group_by(CourseSession.subfolder)
        .order_by(desc("n"), asc("first"))
        .limit(1)
    )
    subfolder_row = (await session.execute(subfolder_stmt)).first()
    canonical_subfolder = subfolder_row.subfolder if subfolder_row else subfolder
    return canonical_folder, canonical_subfolder


async def move_to_folder(
    session: AsyncSession, session_id: uuid.UUID, *, folder: str, subfolder: str
) -> CourseSession | None:
    """Déplace un cours vers un dossier/sous-dossier (créé implicitement s'il n'existe pas encore ;
    réutilise la casse existante d'un dossier/sous-dossier proche, voir `resolve_folder_casing`).

    None si la session est introuvable.
    """
    row = await get_by_id(session, session_id)
    if row is None:
        return None
    resolved_folder, resolved_subfolder = await resolve_folder_casing(session, folder, subfolder)
    row.folder = resolved_folder
    row.subfolder = resolved_subfolder or DEFAULT_COURSE_SUBFOLDER
    await session.commit()
    await session.refresh(row)
    return row


async def delete_folder(session: AsyncSession, folder: str) -> int:
    """Réaffecte tous les cours d'un dossier supprimé vers le dossier par défaut (et son sous-dossier
    par défaut, l'ancien sous-dossier n'ayant plus de sens hors de ce dossier). Retourne le nombre
    de cours déplacés."""
    result = await session.execute(
        update(CourseSession)
        .where(func.lower(CourseSession.folder) == folder.lower())
        .values(folder=DEFAULT_COURSE_FOLDER, subfolder=DEFAULT_COURSE_SUBFOLDER)
    )
    await session.commit()
    return result.rowcount or 0


async def delete_subfolder(session: AsyncSession, folder: str, subfolder: str) -> int:
    """Réaffecte tous les cours d'un sous-dossier supprimé vers le sous-dossier par défaut, dans le
    même dossier. Retourne le nombre de cours déplacés."""
    result = await session.execute(
        update(CourseSession)
        .where(
            func.lower(CourseSession.folder) == folder.lower(),
            func.lower(CourseSession.subfolder) == subfolder.lower(),
        )
        .values(subfolder=DEFAULT_COURSE_SUBFOLDER)
    )
    await session.commit()
    return result.rowcount or 0


async def get_by_id(
    session: AsyncSession,
    session_id: uuid.UUID,
) -> CourseSession | None:
    """Récupère une session par son UUID."""
    result = await session.execute(
        select(CourseSession).where(CourseSession.id == session_id)
    )
    return result.scalar_one_or_none()


async def delete(session: AsyncSession, session_id: uuid.UUID) -> bool:
    """Supprime une session de cours (les jobs podcast, révisions de flashcards et notes de
    section/vidéo liés sont supprimés en cascade par les FK `ON DELETE CASCADE`).

    Retourne False si la session est déjà absente.
    """
    row = await get_by_id(session, session_id)
    if row is None:
        return False
    await session.delete(row)
    await session.commit()
    return True


async def update_section(
    session: AsyncSession,
    session_id: uuid.UUID,
    section_id: str,
    updated_section: dict[str, Any],
) -> CourseSession | None:
    """Remplace une section (par `id`) dans `gemini_response.sections`.

    None si la session ou la section est introuvable. Réassigne tout le dict `gemini_response`
    (plutôt qu'une mutation en place) : `Mapped[dict]` sans `MutableDict` ne détecterait pas une
    mutation en place et ne la persisterait pas.
    """
    row = await get_by_id(session, session_id)
    if row is None:
        return None
    sections = row.gemini_response.get("sections") or []
    index = next((i for i, s in enumerate(sections) if str(s.get("id")) == section_id), None)
    if index is None:
        return None
    row.gemini_response = {
        **row.gemini_response,
        "sections": [*sections[:index], updated_section, *sections[index + 1 :]],
    }
    await session.commit()
    await session.refresh(row)
    return row
