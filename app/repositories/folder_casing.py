"""Résolution de la casse canonique d'un dossier/sous-dossier, commune aux cours et aux fichiers."""

from sqlalchemy import asc, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import InstrumentedAttribute


async def resolve_folder_casing(
    session: AsyncSession,
    *,
    folder_column: InstrumentedAttribute[str],
    subfolder_column: InstrumentedAttribute[str],
    age_column: InstrumentedAttribute,
    folder: str,
    subfolder: str | None = None,
) -> tuple[str, str | None]:
    """Résout la casse canonique d'un dossier (et d'un sous-dossier) déjà utilisés dans la table
    des colonnes fournies, par comparaison insensible à la casse — pour que « sfi » tapé quand
    « Sfi » existe déjà rejoigne « Sfi » au lieu de créer un dossier distinct.

    Casse canonique = celle portée par le plus de lignes (égalité : la plus ancienne selon
    `age_column`). Retourne la valeur fournie telle quelle si aucune variante n'existe encore.
    """
    folder_stmt = (
        select(folder_column.label("folder"), func.count().label("n"), func.min(age_column).label("first"))
        .where(func.lower(folder_column) == folder.lower())
        .group_by(folder_column)
        .order_by(desc("n"), asc("first"))
        .limit(1)
    )
    folder_row = (await session.execute(folder_stmt)).first()
    canonical_folder = folder_row.folder if folder_row else folder

    if not subfolder:
        return canonical_folder, subfolder

    subfolder_stmt = (
        select(subfolder_column.label("subfolder"), func.count().label("n"), func.min(age_column).label("first"))
        .where(
            func.lower(folder_column) == canonical_folder.lower(),
            func.lower(subfolder_column) == subfolder.lower(),
        )
        .group_by(subfolder_column)
        .order_by(desc("n"), asc("first"))
        .limit(1)
    )
    subfolder_row = (await session.execute(subfolder_stmt)).first()
    canonical_subfolder = subfolder_row.subfolder if subfolder_row else subfolder
    return canonical_folder, canonical_subfolder
