"""Tests unitaires pour extract_pdf_chunks() : construction des chunks et délégation aux threads."""

from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.services import pdf_pipeline


@pytest.fixture
def settings() -> Settings:
    return Settings(gemini_api_key="fake-key")


@pytest.mark.asyncio
async def test_extract_pdf_chunks_builds_one_chunk_per_short_page(settings, monkeypatch):
    monkeypatch.setattr(pdf_pipeline, "_extract_text_from_pdf", lambda _: ["Page un.", "Page deux."])
    monkeypatch.setattr(pdf_pipeline, "_extract_tables_from_pdf", lambda _: [])
    monkeypatch.setattr(pdf_pipeline, "extract_key_image_descriptions", AsyncMock(return_value=[]))

    chunks = await pdf_pipeline.extract_pdf_chunks(b"%PDF-1.4", "doc.pdf", settings)

    assert [c["content"] for c in chunks] == ["Page un.", "Page deux."]
    assert [c["metadata"]["page"] for c in chunks] == [1, 2]
    assert all(c["metadata"]["filename"] == "doc.pdf" for c in chunks)


@pytest.mark.asyncio
async def test_extract_pdf_chunks_combines_text_tables_and_images(settings, monkeypatch):
    monkeypatch.setattr(pdf_pipeline, "_extract_text_from_pdf", lambda _: ["Texte"])
    monkeypatch.setattr(pdf_pipeline, "_extract_tables_from_pdf", lambda _: ["Table page 1:\n| a |"])
    monkeypatch.setattr(
        pdf_pipeline, "extract_key_image_descriptions", AsyncMock(return_value=["Description image"])
    )

    chunks = await pdf_pipeline.extract_pdf_chunks(b"%PDF-1.4", "doc.pdf", settings)

    # `chunk_text` normalise les espaces (y compris les sauts de ligne) en un seul espace.
    assert [c["content"] for c in chunks] == ["Texte", "Table page 1: | a |", "Description image"]


@pytest.mark.asyncio
async def test_extract_pdf_chunks_offloads_blocking_extraction_to_a_thread(settings, monkeypatch):
    """`fitz`/`camelot` sont synchrones : ils doivent tourner via `asyncio.to_thread`, jamais
    directement dans la coroutine, sous peine de bloquer tout le serveur sur un gros PDF."""
    import threading

    caller_thread = threading.current_thread()
    seen_threads: list[threading.Thread] = []

    def _fake_extract_text(_: bytes) -> list[str]:
        seen_threads.append(threading.current_thread())
        return ["Texte"]

    monkeypatch.setattr(pdf_pipeline, "_extract_text_from_pdf", _fake_extract_text)
    monkeypatch.setattr(pdf_pipeline, "_extract_tables_from_pdf", lambda _: [])
    monkeypatch.setattr(pdf_pipeline, "extract_key_image_descriptions", AsyncMock(return_value=[]))

    await pdf_pipeline.extract_pdf_chunks(b"%PDF-1.4", "doc.pdf", settings)

    assert seen_threads and seen_threads[0] is not caller_thread


def test_extract_tables_from_pdf_returns_empty_on_camelot_failure(monkeypatch):
    """Une page illisible par camelot (image scannée, page corrompue) ne doit jamais faire
    échouer toute l'ingestion : repli silencieux sur aucune table."""
    def _boom(*_args, **_kwargs):
        raise RuntimeError("camelot native crash")

    monkeypatch.setattr(pdf_pipeline, "camelot", type("_Fake", (), {"read_pdf": staticmethod(_boom)}))

    assert pdf_pipeline._extract_tables_from_pdf(b"%PDF-1.4") == []
