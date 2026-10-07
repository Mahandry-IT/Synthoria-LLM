"""Rangement des fichiers ingérés en dossiers/sous-dossiers (table `ingested_files`).

Un fichier sans ligne est rangé dans le dossier par défaut : seules les lignes des fichiers
explicitement déplacés (ou ingérés directement dans un dossier) existent. `filename` n'est jamais
utilisé autrement que comme clé de recherche exacte.
"""

from collections.abc import Iterable

from sqlalchemy import delete as sql_delete
from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import DEFAULT_COURSE_FOLDER, DEFAULT_COURSE_SUBFOLDER, IngestedFile
from app.repositories import folder_casing


async def get_placements(session: AsyncSession, filenames: Iterable[str]) -> dict[str, tuple[str, str]]:
    """`{filename: (folder, subfolder)}` des fichiers donnés ayant une ligne, en une seule requête.

    Les fichiers absents du résultat sont dans le dossier par défaut.
    """
    names = list(filenames)
    if not names:
        return {}
    result = await session.execute(
        select(IngestedFile.filename, IngestedFile.folder, IngestedFile.subfolder)
        .where(IngestedFile.filename.in_(names))
    )
    return {filename: (folder, subfolder) for filename, folder, subfolder in result.all()}


async def resolve_casing(
    session: AsyncSession, folder: str, subfolder: str | None = None
) -> tuple[str, str | None]:
    """Casse canonique d'un dossier/sous-dossier de fichiers existant (le plus de fichiers, puis
    le plus ancien) ; valeur fournie telle quelle s'il est nouveau."""
    return await folder_casing.resolve_folder_casing(
        session,
        folder_column=IngestedFile.folder,
        subfolder_column=IngestedFile.subfolder,
        age_column=IngestedFile.updated_at,
        folder=folder,
        subfolder=subfolder,
    )


async def move_many(
    session: AsyncSession, filenames: Iterable[str], *, folder: str, subfolder: str | None
) -> tuple[str, str]:
    """Range des fichiers dans un dossier/sous-dossier (créés implicitement, casse existante
    réutilisée). Upsert idempotent en une seule requête. Retourne `(folder, subfolder)` résolus.

    L'appelant vérifie que les fichiers existent dans le vector store (pas de ligne orpheline).
    """
    resolved_folder, resolved_subfolder = await resolve_casing(session, folder, subfolder)
    resolved_subfolder = resolved_subfolder or DEFAULT_COURSE_SUBFOLDER
    rows = [
        {"filename": name, "folder": resolved_folder, "subfolder": resolved_subfolder}
        for name in dict.fromkeys(filenames)
    ]
    if rows:
        stmt = insert(IngestedFile).values(rows)
        await session.execute(
            stmt.on_conflict_do_update(
                index_elements=[IngestedFile.filename],
                set_={"folder": stmt.excluded.folder, "subfolder": stmt.excluded.subfolder, "updated_at": func.now()},
            )
        )
        await session.commit()
    return resolved_folder, resolved_subfolder


async def move(
    session: AsyncSession, filename: str, *, folder: str, subfolder: str | None
) -> tuple[str, str]:
    """Range un fichier dans un dossier/sous-dossier — voir `move_many`."""
    return await move_many(session, [filename], folder=folder, subfolder=subfolder)


async def delete(session: AsyncSession, filename: str) -> None:
    """Supprime la ligne de rangement d'un fichier (no-op si absente)."""
    await session.execute(sql_delete(IngestedFile).where(IngestedFile.filename == filename))
    await session.commit()


async def delete_folder(session: AsyncSession, folder: str) -> int:
    """Reclasse les fichiers d'un dossier (tous sous-dossiers confondus) au dossier et sous-dossier
    par défaut — aucun fichier n'est supprimé. Retourne le nombre de fichiers reclassés."""
    result = await session.execute(
        update(IngestedFile)
        .where(func.lower(IngestedFile.folder) == folder.lower())
        .values(folder=DEFAULT_COURSE_FOLDER, subfolder=DEFAULT_COURSE_SUBFOLDER, updated_at=func.now())
    )
    await session.commit()
    return result.rowcount or 0


async def delete_subfolder(session: AsyncSession, folder: str, subfolder: str) -> int:
    """Reclasse les fichiers d'un sous-dossier au sous-dossier par défaut du même dossier.
    Retourne le nombre de fichiers reclassés."""
    result = await session.execute(
        update(IngestedFile)
        .where(
            func.lower(IngestedFile.folder) == folder.lower(),
            func.lower(IngestedFile.subfolder) == subfolder.lower(),
        )
        .values(subfolder=DEFAULT_COURSE_SUBFOLDER, updated_at=func.now())
    )
    await session.commit()
    return result.rowcount or 0
