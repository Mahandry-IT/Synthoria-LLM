"""Figures des PDF ingérés (table `ingested_file_figures`), voir `app.services.media.pdf_figures`."""

import uuid

from sqlalchemy import delete as sql_delete
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import IngestedFileFigure, MediaAsset


async def add(session: AsyncSession, *, filename: str, ref: str, page: int, asset_id: uuid.UUID) -> None:
    """Relie une figure à son fichier ; no-op si ce fichier la référence déjà (ré-ingestion)."""
    await session.execute(
        insert(IngestedFileFigure)
        .values(filename=filename, ref=ref, page=page, asset_id=asset_id)
        .on_conflict_do_nothing(constraint="uq_ingested_file_figures_filename_ref")
    )
    await session.commit()


async def get_asset_by_ref(session: AsyncSession, ref: str) -> MediaAsset | None:
    """Image d'une figure encore reliée à un fichier ingéré ; None si la référence est inconnue.

    Une même image présente dans plusieurs fichiers porte la même référence et le même asset.
    """
    result = await session.execute(
        select(MediaAsset)
        .join(IngestedFileFigure, IngestedFileFigure.asset_id == MediaAsset.id)
        .where(IngestedFileFigure.ref == ref)
        .limit(1)
    )
    return result.scalars().first()


async def delete_by_filename(session: AsyncSession, filename: str) -> None:
    """Délie les figures d'un fichier supprimé (les images restent : des cours les affichent)."""
    await session.execute(sql_delete(IngestedFileFigure).where(IngestedFileFigure.filename == filename))
    await session.commit()
