"""Normalisation et stockage des images (ré-hébergement systématique, jamais de hotlink).

Sécurité : ré-encodage systématique (l'octet source n'est jamais servi tel quel), format vérifié
par signature Pillow (jamais par le Content-Type déclaré par la source), taille bornée avant même
le décodage, bombe de décompression bornée (`Image.MAX_IMAGE_PIXELS`), métadonnées EXIF retirées
par le ré-encodage, dimension maximale.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

from app.core.config import Settings

# Bombe de décompression (PIL.Image.DecompressionBombError au-delà) : ~40 Mpx, largement
# suffisant pour une figure ou une photo, bloque les images délibérément gigantesques.
Image.MAX_IMAGE_PIXELS = 40_000_000

_DEFAULT_MAX_DIMENSION = 1600
_SAVE_FORMAT = "WEBP"
_MIME = "image/webp"


class UnsupportedImageError(Exception):
    """`raw` n'est pas une image exploitable : signature invalide, format non supporté, trop grande."""


@dataclass(frozen=True)
class NormalizedImage:
    content: bytes
    sha256: str
    mime: str
    width: int
    height: int


def normalize_image(raw: bytes, *, settings: Settings, max_dimension: int = _DEFAULT_MAX_DIMENSION) -> NormalizedImage:
    """Ré-encode `raw` en WebP : EXIF appliqué puis retiré, redimensionnée si besoin.

    Lève `UnsupportedImageError` si `raw` dépasse `settings.media_max_bytes`, n'est pas une image
    valide, ou dépasse `Image.MAX_IMAGE_PIXELS` (bombe de décompression) — jamais d'autre exception.
    """
    if len(raw) > settings.media_max_bytes:
        raise UnsupportedImageError(f"Image trop volumineuse : {len(raw)} octets > {settings.media_max_bytes}")

    try:
        with Image.open(io.BytesIO(raw)) as opened:
            opened.load()  # force la lecture complète : une image tronquée échoue ici, pas plus tard
            image = ImageOps.exif_transpose(opened) or opened
            if image.mode not in ("RGB", "RGBA"):
                image = image.convert("RGB")
    except UnsupportedImageError:
        raise
    except Exception as exc:  # Pillow lève des types variés (UnidentifiedImageError, DecompressionBombError, OSError...)
        raise UnsupportedImageError(f"Image invalide ou non supportée : {exc}") from exc

    image.thumbnail((max_dimension, max_dimension), Image.LANCZOS)  # ne redimensionne qu'à la baisse

    buffer = io.BytesIO()
    image.save(buffer, format=_SAVE_FORMAT, quality=85, method=6)  # ré-encodage : EXIF jamais recopié
    content = buffer.getvalue()
    return NormalizedImage(
        content=content,
        sha256=hashlib.sha256(content).hexdigest(),
        mime=_MIME,
        width=image.width,
        height=image.height,
    )


def storage_path_for(settings: Settings, sha256: str) -> Path:
    """Chemin déterministe depuis le sha256 (jamais depuis une entrée utilisateur)."""
    return Path(settings.media_storage_dir) / sha256[:2] / f"{sha256}.webp"


def write_to_disk(settings: Settings, image: NormalizedImage) -> Path:
    """Écrit le fichier normalisé sur disque ; no-op si déjà présent (même sha256 = même contenu)."""
    path = storage_path_for(settings, image.sha256)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(image.content)
    return path
