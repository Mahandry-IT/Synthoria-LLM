import io

import pytest
from PIL import Image

from app.core.config import Settings
from app.services.media.storage import (
    UnsupportedImageError,
    normalize_image,
    storage_path_for,
    write_to_disk,
)


def _png_bytes(size: tuple[int, int] = (50, 40), color: str = "red") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_normalize_image_reencodes_to_webp_and_hashes_content():
    settings = Settings(media_max_bytes=5_000_000)
    result = normalize_image(_png_bytes(), settings=settings)
    assert result.mime == "image/webp"
    assert (result.width, result.height) == (50, 40)
    assert len(result.sha256) == 64
    assert Image.open(io.BytesIO(result.content)).format == "WEBP"


def test_normalize_image_downscales_beyond_max_dimension():
    settings = Settings(media_max_bytes=5_000_000)
    result = normalize_image(_png_bytes((3000, 100)), settings=settings, max_dimension=1600)
    assert result.width == 1600
    assert result.height < 100  # ratio conservé, jamais agrandi au-delà de l'original en hauteur


def test_normalize_image_strips_exif():
    buffer = io.BytesIO()
    image = Image.new("RGB", (20, 20), color="blue")
    exif = image.getexif()
    exif[0x0112] = 3  # Orientation : présence vérifiable, doit disparaître au ré-encodage
    image.save(buffer, format="JPEG", exif=exif)

    settings = Settings(media_max_bytes=5_000_000)
    result = normalize_image(buffer.getvalue(), settings=settings)
    reencoded = Image.open(io.BytesIO(result.content))
    assert not reencoded.getexif()


def test_normalize_image_rejects_oversized_payload():
    settings = Settings(media_max_bytes=10)
    with pytest.raises(UnsupportedImageError):
        normalize_image(_png_bytes(), settings=settings)


def test_normalize_image_rejects_fake_mime_content():
    """Le contenu n'est jamais cru sur parole : un faux fichier (peu importe son Content-Type déclaré) échoue ici."""
    settings = Settings(media_max_bytes=5_000_000)
    with pytest.raises(UnsupportedImageError):
        normalize_image(b"not actually an image, just plain bytes claiming to be one", settings=settings)


def test_normalize_image_rejects_decompression_bomb():
    settings = Settings(media_max_bytes=200_000_000)
    huge = _png_bytes((10000, 10000))  # > Image.MAX_IMAGE_PIXELS (40M px), quel que soit media_max_bytes
    with pytest.raises(UnsupportedImageError):
        normalize_image(huge, settings=settings)


def test_storage_path_for_is_deterministic_from_sha256():
    settings = Settings(media_storage_dir="/data/media")
    sha = "abcd" * 16
    path = storage_path_for(settings, sha)
    assert path.name == f"{sha}.webp"
    assert path.parent.name == sha[:2]
    # Même sha256 → même chemin (jamais dépendant d'une entrée utilisateur ou d'un timestamp).
    assert storage_path_for(settings, sha) == path


def test_write_to_disk_is_idempotent(tmp_path):
    settings = Settings(media_storage_dir=str(tmp_path), media_max_bytes=5_000_000)
    image = normalize_image(_png_bytes(), settings=settings)
    path1 = write_to_disk(settings, image)
    assert path1.is_file()
    content_before = path1.read_bytes()
    path2 = write_to_disk(settings, image)
    assert path2 == path1
    assert path1.read_bytes() == content_before
