"""Tests unitaires pour NumpyVectorStore.remove_file()."""

import json
from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.services.vector_store import NumpyVectorStore


@pytest.fixture
def ollama_client():
    return AsyncMock()


@pytest.fixture
def store(tmp_path, ollama_client) -> NumpyVectorStore:
    settings = Settings(gemini_api_key="fake-key", chroma_persist_directory=str(tmp_path))
    return NumpyVectorStore(settings=settings, ollama_client=ollama_client)


def _doc(filename: str, page: int) -> dict:
    return {
        "id": f"{filename}::{page}",
        "content": f"page {page}",
        "metadata": {"filename": filename, "page": page},
        "embedding": [0.1],
    }


def test_remove_file_deletes_matching_chunks_and_returns_count(store):
    store._documents = [_doc("doc.pdf", 1), _doc("doc.pdf", 2), _doc("other.pdf", 1)]

    removed = store.remove_file("doc.pdf")

    assert removed == 2
    assert store._documents == [_doc("other.pdf", 1)]
    assert store.has_file("doc.pdf") is False
    assert store.has_file("other.pdf") is True


def test_remove_file_persists_to_disk(store):
    store._documents = [_doc("doc.pdf", 1)]

    store.remove_file("doc.pdf")

    assert store._index_path.exists()
    assert json.loads(store._index_path.read_text(encoding="utf-8")) == []


def test_remove_file_unknown_filename_returns_zero_and_does_not_write(store):
    store._documents = [_doc("doc.pdf", 1)]

    removed = store.remove_file("unknown.pdf")

    assert removed == 0
    assert store._documents == [_doc("doc.pdf", 1)]
    assert not store._index_path.exists()


def test_remove_file_empty_store_returns_zero(store):
    assert store.remove_file("doc.pdf") == 0
