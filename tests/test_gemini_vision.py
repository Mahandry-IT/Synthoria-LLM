import io
from unittest.mock import AsyncMock

import fitz
import pytest
from PIL import Image

from app.core.exceptions import GeminiQuotaExceededError
from app.services.gemini_vision import extract_key_image_descriptions

_LARGE_SIZE = 60  # au-dessus du seuil _MIN_IMAGE_DIMENSION_PX (50px)
_SMALL_SIZE = 10  # en dessous, doit être écarté par le préfiltre


def _pdf_with_images(images_per_page: int = 1, pages: int = 1, size: int = _LARGE_SIZE) -> bytes:
    """Chaque image a une couleur distincte (par page/position) pour ne jamais collisionner
    accidentellement avec le dédoublonnage par hash — sauf demande explicite (voir tests dédiés)."""
    doc = fitz.open()
    for page_num in range(pages):
        page = doc.new_page()
        for i in range(images_per_page):
            buf = io.BytesIO()
            color = (page_num * 40 % 256, i * 60 % 256, 128)
            Image.new("RGB", (size, size), color=color).save(buf, format="PNG")
            page.insert_image(fitz.Rect(i * 20, 0, i * 20 + 15, 15), stream=buf.getvalue())
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def _pdf_without_images() -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((10, 10), "Texte sans image.")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def _items(*, informative: bool = True, description: str = "desc", count: int = 1) -> dict:
    return {"items": [{"index": i, "informative": informative, "description": description} for i in range(count)]}


@pytest.mark.asyncio
async def test_returns_empty_when_no_gemini_client():
    assert await extract_key_image_descriptions(_pdf_with_images(), None) == []


@pytest.mark.asyncio
async def test_returns_empty_when_gemini_client_not_configured():
    client = AsyncMock()
    client.is_configured = False
    assert await extract_key_image_descriptions(_pdf_with_images(), client) == []
    client.describe_images.assert_not_awaited()


@pytest.mark.asyncio
async def test_returns_empty_for_pdf_without_images():
    client = AsyncMock()
    client.is_configured = True
    assert await extract_key_image_descriptions(_pdf_without_images(), client) == []
    client.describe_images.assert_not_awaited()


@pytest.mark.asyncio
async def test_describes_images_via_a_single_batched_call():
    client = AsyncMock()
    client.is_configured = True
    client.describe_images.return_value = {
        "items": [
            {"index": 0, "informative": True, "description": "Un graphique montrant une croissance."},
            {"index": 1, "informative": True, "description": "Un second graphique."},
        ]
    }

    descriptions = await extract_key_image_descriptions(_pdf_with_images(images_per_page=2), client)

    client.describe_images.assert_awaited_once()  # 1 seul appel réseau pour les 2 images
    assert len(descriptions) == 2
    assert "Un graphique montrant une croissance." in descriptions[0]
    assert "page 1" in descriptions[0]


@pytest.mark.asyncio
async def test_skips_non_informative_image_without_failing():
    client = AsyncMock()
    client.is_configured = True
    client.describe_images.return_value = _items(informative=False, description="")

    assert await extract_key_image_descriptions(_pdf_with_images(), client) == []


@pytest.mark.asyncio
async def test_one_malformed_item_does_not_discard_the_others_in_the_same_batch():
    """Un item non conforme au schéma (ex. description trop longue) pour UNE image ne doit pas
    faire perdre les descriptions des autres images du même appel groupé."""
    client = AsyncMock()
    client.is_configured = True
    client.describe_images.return_value = {
        "items": [
            {"index": 0, "informative": True, "description": "x" * 5000},  # dépasse max_length=2000
            {"index": 1, "informative": True, "description": "Description valide."},
        ]
    }

    descriptions = await extract_key_image_descriptions(_pdf_with_images(images_per_page=2), client)

    assert len(descriptions) == 1
    assert "Description valide." in descriptions[0]


@pytest.mark.asyncio
async def test_duplicate_index_in_response_only_uses_the_first_occurrence():
    client = AsyncMock()
    client.is_configured = True
    client.describe_images.return_value = {
        "items": [
            {"index": 0, "informative": True, "description": "Première description."},
            {"index": 0, "informative": True, "description": "Deuxième description (doublon)."},
        ]
    }

    descriptions = await extract_key_image_descriptions(_pdf_with_images(), client)

    assert len(descriptions) == 1
    assert "Première description." in descriptions[0]


@pytest.mark.asyncio
async def test_best_effort_returns_empty_when_batched_call_fails():
    """L'appel groupé échoue (quota, indisponibilité) : toutes les images de ce PDF sont ignorées,
    jamais fatal pour l'ingestion du PDF."""
    client = AsyncMock()
    client.is_configured = True
    client.describe_images.side_effect = GeminiQuotaExceededError("quota dépassé")

    descriptions = await extract_key_image_descriptions(_pdf_with_images(images_per_page=2), client)

    assert descriptions == []
    client.describe_images.assert_awaited_once()


@pytest.mark.asyncio
async def test_caps_pages_and_images_per_page_within_a_single_call():
    """Au plus 5 pages, au plus 2 images par page — borne le coût même sur un gros PDF, mais un
    seul appel réseau au total (pas un par image)."""
    client = AsyncMock()
    client.is_configured = True
    client.describe_images.return_value = _items(description="desc", count=10)

    descriptions = await extract_key_image_descriptions(_pdf_with_images(images_per_page=3, pages=7), client)

    client.describe_images.assert_awaited_once()
    sent_images = client.describe_images.await_args.args[0]
    assert len(sent_images) == 5 * 2
    assert len(descriptions) == 10


@pytest.mark.asyncio
async def test_filters_out_undersized_images_before_calling_gemini():
    """Une image trop petite (icône/puce décorative) est écartée localement, sans appel Gemini."""
    client = AsyncMock()
    client.is_configured = True

    assert await extract_key_image_descriptions(_pdf_with_images(size=_SMALL_SIZE), client) == []
    client.describe_images.assert_not_awaited()


@pytest.mark.asyncio
async def test_deduplicates_identical_images_by_hash_before_calling_gemini():
    """Deux images strictement identiques (même bytes) ne sont envoyées qu'une seule fois."""
    doc = fitz.open()
    buf = io.BytesIO()
    Image.new("RGB", (_LARGE_SIZE, _LARGE_SIZE), color="red").save(buf, format="PNG")
    same_image_bytes = buf.getvalue()
    for _ in range(2):
        page = doc.new_page()
        page.insert_image(fitz.Rect(0, 0, 15, 15), stream=same_image_bytes)
    pdf_bytes = doc.tobytes()
    doc.close()

    client = AsyncMock()
    client.is_configured = True
    client.describe_images.return_value = _items(description="desc", count=1)

    await extract_key_image_descriptions(pdf_bytes, client)

    client.describe_images.assert_awaited_once()
    sent_images = client.describe_images.await_args.args[0]
    assert len(sent_images) == 1
