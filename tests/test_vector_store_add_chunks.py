"""Tests unitaires pour NumpyVectorStore.add_chunks() : atomicité et concurrence bornée."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.core.exceptions import OllamaUnavailableError
from app.services.vector_store import NumpyVectorStore


@pytest.fixture
def ollama_client():
    return AsyncMock()


@pytest.fixture
def store(tmp_path, ollama_client) -> NumpyVectorStore:
    settings = Settings(gemini_api_key="fake-key", chroma_persist_directory=str(tmp_path))
    return NumpyVectorStore(settings=settings, ollama_client=ollama_client)


def _chunk(i: int) -> dict:
    return {"id": f"doc::{i}", "content": f"chunk {i}", "metadata": {"filename": "doc.pdf", "page": i}}


@pytest.mark.asyncio
async def test_add_chunks_persists_all_embeddings(store, ollama_client):
    ollama_client.embed = AsyncMock(return_value=[0.1, 0.2])
    chunks = [_chunk(i) for i in range(5)]

    added = await store.add_chunks(chunks)

    assert added == 5
    assert len(store._documents) == 5
    assert store.has_file("doc.pdf")
    assert store._index_path.exists()


@pytest.mark.asyncio
async def test_add_chunks_leaves_no_partial_state_on_failure(store, ollama_client):
    """Un échec en cours de route ne doit laisser ni chunk en mémoire ni fichier sur disque : sinon
    `has_file` verrait le document comme déjà indexé (état fantôme) sans rien de réellement
    persisté, bloquant tout nouvel essai jusqu'au redémarrage du process."""
    ollama_client.embed = AsyncMock(side_effect=[[0.1], OllamaUnavailableError("down"), [0.1]])
    chunks = [_chunk(i) for i in range(3)]

    with pytest.raises(OllamaUnavailableError):
        await store.add_chunks(chunks)

    assert store._documents == []
    assert store.has_file("doc.pdf") is False
    assert not store._index_path.exists()


@pytest.mark.asyncio
async def test_add_chunks_respects_bounded_concurrency(store, ollama_client):
    """N'a jamais plus de `pdf_embedding_concurrency` appels d'embedding en vol simultanément —
    condition nécessaire pour accélérer un gros PDF sans saturer Ollama."""
    in_flight = 0
    max_in_flight = 0

    async def fake_embed(text: str) -> list[float]:
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return [0.1]

    ollama_client.embed = fake_embed
    chunks = [_chunk(i) for i in range(20)]

    await store.add_chunks(chunks)

    assert max_in_flight == store._settings.pdf_embedding_concurrency
