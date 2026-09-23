import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import MediaAsset


async def get(session: AsyncSession, asset_id: uuid.UUID) -> MediaAsset | None:
    return await session.get(MediaAsset, asset_id)


async def get_by_sha256(session: AsyncSession, sha256: str) -> MediaAsset | None:
    """Déduplication : deux blocs qui résolvent vers le même contenu partagent une seule ligne."""
    result = await session.execute(select(MediaAsset).where(MediaAsset.sha256 == sha256))
    return result.scalar_one_or_none()


async def create(
    session: AsyncSession,
    *,
    kind: str,
    sha256: str,
    mime: str,
    width: int,
    height: int,
    size_bytes: int,
    storage_path: str,
    alt_text: str = "",
    origin_url: str | None = None,
    author: str | None = None,
    license: str | None = None,
    license_url: str | None = None,
    source_filename: str | None = None,
    source_page: int | None = None,
) -> MediaAsset:
    asset = MediaAsset(
        kind=kind, sha256=sha256, mime=mime, width=width, height=height, bytes=size_bytes,
        storage_path=storage_path, alt_text=alt_text, origin_url=origin_url, author=author,
        license=license, license_url=license_url, source_filename=source_filename, source_page=source_page,
    )
    session.add(asset)
    await session.commit()
    await session.refresh(asset)
    return asset


async def get_or_create(
    session: AsyncSession,
    *,
    sha256: str,
    **create_kwargs,
) -> MediaAsset:
    """Réutilise la ligne existante si le contenu (sha256) est déjà connu, sinon en crée une."""
    existing = await get_by_sha256(session, sha256)
    if existing is not None:
        return existing
    return await create(session, sha256=sha256, **create_kwargs)
