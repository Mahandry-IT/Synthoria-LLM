import io
from unittest.mock import AsyncMock

import fitz
import pytest
from PIL import Image

from app.core.exceptions import GeminiQuotaExceededError
from app.services.gemini_vision import extract_key_image_descriptions


def _pdf_with_images(images_per_page: int = 1, pages: int = 1) -> bytes:
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page()
        for i in range(images_per_page):
            buf = io.BytesIO()
            Image.new("RGB", (10, 10), color="red").save(buf, format="PNG")
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


@pytest.mark.asyncio
async def test_returns_empty_when_no_gemini_client():
    assert await extract_key_image_descriptions(_pdf_with_images(), None) == []


@pytest.mark.asyncio
async def test_returns_empty_when_gemini_client_not_configured():
    client = AsyncMock()
    client.is_configured = False
    assert await extract_key_image_descriptions(_pdf_with_images(), client) == []
    client.describe_image.assert_not_awaited()


@pytest.mark.asyncio
async def test_returns_empty_for_pdf_without_images():
    client = AsyncMock()
    client.is_configured = True
    assert await extract_key_image_descriptions(_pdf_without_images(), client) == []
    client.describe_image.assert_not_awaited()


@pytest.mark.asyncio
async def test_describes_each_image_via_gemini_client():
    client = AsyncMock()
    client.is_configured = True
    client.describe_image.return_value = "Un graphique montrant une croissance."

    descriptions = await extract_key_image_descriptions(_pdf_with_images(images_per_page=2), client)

    assert client.describe_image.await_count == 2
    assert len(descriptions) == 2
    assert "Un graphique montrant une croissance." in descriptions[0]
    assert "page 1" in descriptions[0]


@pytest.mark.asyncio
async def test_skips_non_informative_image_without_failing():
    client = AsyncMock()
    client.is_configured = True
    client.describe_image.return_value = ""  # image jugée non informative par les instructions

    assert await extract_key_image_descriptions(_pdf_with_images(), client) == []


@pytest.mark.asyncio
async def test_best_effort_skips_image_on_gemini_error_without_raising():
    """Une image dont la description échoue (quota, indisponibilité) est ignorée, jamais fatale."""
    client = AsyncMock()
    client.is_configured = True
    client.describe_image.side_effect = [
        GeminiQuotaExceededError("quota dépassé"),
        "Deuxième image, décrite normalement.",
    ]

    descriptions = await extract_key_image_descriptions(_pdf_with_images(images_per_page=2), client)

    assert len(descriptions) == 1
    assert "Deuxième image" in descriptions[0]


@pytest.mark.asyncio
async def test_caps_pages_and_images_per_page():
    """Au plus 5 pages, au plus 2 images par page — borne le coût même sur un gros PDF."""
    client = AsyncMock()
    client.is_configured = True
    client.describe_image.return_value = "desc"

    descriptions = await extract_key_image_descriptions(_pdf_with_images(images_per_page=3, pages=7), client)

    assert client.describe_image.await_count == 5 * 2
    assert len(descriptions) == 10
