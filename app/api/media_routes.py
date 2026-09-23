"""Route de service des images ré-hébergées : `GET /media/{asset_id}`, jamais de hotlink.

`asset_id` est validé comme UUID par FastAPI avant même d'atteindre la fonction ; le chemin sur
disque est toujours reconstruit depuis la base (`storage_path_for`), jamais depuis une entrée
utilisateur — un `asset_id` valide mais inconnu retourne 404 sans toucher le système de fichiers
avec une entrée non vérifiée.
"""

from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import FileResponse, Response
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.core.config import Settings, get_settings
from app.repositories import media_asset_repository
from app.services.media.storage import storage_path_for

router = APIRouter(tags=["media"])


def _session_factory(request: Request) -> async_sessionmaker:
    return request.app.state.db_session_factory


@router.get("/media/{asset_id}")
async def get_media_asset(
    request: Request, asset_id: UUID, settings: Settings = Depends(get_settings)
) -> Response:
    async with _session_factory(request)() as db:
        asset = await media_asset_repository.get(db, asset_id)
    if asset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image introuvable")

    path: Path = storage_path_for(settings, asset.sha256)
    if not path.is_file():
        raise HTTPException(status_code=status.HTTP_410_GONE, detail="Fichier expiré ou supprimé")

    return FileResponse(
        path,
        media_type=asset.mime,
        headers={
            "Cache-Control": "public, max-age=31536000, immutable",
            "X-Content-Type-Options": "nosniff",
        },
    )
