"""Tests unitaires pour list_files() et has_file() de NumpyVectorStore."""

from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.services.vector_store import NumpyVectorStore


@pytest.fixture
def settings() -> Settings:
    return Settings(gemini_api_key="fake-key")


@pytest.fixture
def ollama_client():
    return AsyncMock()


def _make_store_with_docs(settings, ollama_client, docs: list[dict]) -> NumpyVectorStore:
    store = NumpyVectorStore(settings=settings, ollama_client=ollama_client)
    store._documents = docs
    return store


# --- Tests list_files ---


def test_list_files_empty(settings, ollama_client):
    """Vector store vide → retourne une liste vide."""
    store = _make_store_with_docs(settings, ollama_client, [])
    result = store.list_files()
    assert result == []


def test_list_files_single_file(settings, ollama_client):
    """Un seul fichier avec plusieurs chunks → retourne un seul fichier avec id=1."""
    docs = [
        {"id": "1", "content": "a", "metadata": {"filename": "doc.pdf", "page": 1}, "embedding": [1.0]},
        {"id": "2", "content": "b", "metadata": {"filename": "doc.pdf", "page": 2}, "embedding": [1.0]},
    ]
    store = _make_store_with_docs(settings, ollama_client, docs)
    result = store.list_files()
    assert result == [{"id": 1, "filename": "doc.pdf"}]


def test_list_files_multiple_files(settings, ollama_client):
    """Deux fichiers → retourne les deux avec ids séquentiels."""
    docs = [
        {"id": "1", "content": "a", "metadata": {"filename": "first.pdf", "page": 1}, "embedding": [1.0]},
        {"id": "2", "content": "b", "metadata": {"filename": "second.pdf", "page": 1}, "embedding": [1.0]},
        {"id": "3", "content": "c", "metadata": {"filename": "first.pdf", "page": 2}, "embedding": [1.0]},
    ]
    store = _make_store_with_docs(settings, ollama_client, docs)
    result = store.list_files()
    assert result == [
        {"id": 1, "filename": "first.pdf"},
        {"id": 2, "filename": "second.pdf"},
    ]


def test_list_files_preserves_insertion_order(settings, ollama_client):
    """L'ID est basé sur l'ordre d'apparition, pas l'ordre alphabétique."""
    docs = [
        {"id": "1", "content": "a", "metadata": {"filename": "zebra.pdf", "page": 1}, "embedding": [1.0]},
        {"id": "2", "content": "b", "metadata": {"filename": "alpha.pdf", "page": 1}, "embedding": [1.0]},
    ]
    store = _make_store_with_docs(settings, ollama_client, docs)
    result = store.list_files()
    assert result[0]["filename"] == "zebra.pdf"
    assert result[0]["id"] == 1
    assert result[1]["filename"] == "alpha.pdf"
    assert result[1]["id"] == 2


# --- Tests has_file ---


def test_has_file_true(settings, ollama_client):
    """Le fichier existe → retourne True."""
    docs = [
        {"id": "1", "content": "a", "metadata": {"filename": "doc.pdf", "page": 1}, "embedding": [1.0]},
    ]
    store = _make_store_with_docs(settings, ollama_client, docs)
    assert store.has_file("doc.pdf") is True


def test_has_file_false(settings, ollama_client):
    """Le fichier n'existe pas → retourne False."""
    docs = [
        {"id": "1", "content": "a", "metadata": {"filename": "doc.pdf", "page": 1}, "embedding": [1.0]},
    ]
    store = _make_store_with_docs(settings, ollama_client, docs)
    assert store.has_file("other.pdf") is False


def test_has_file_empty_store(settings, ollama_client):
    """Vector store vide → retourne False."""
    store = _make_store_with_docs(settings, ollama_client, [])
    assert store.has_file("doc.pdf") is False


# --- Tests get_all_chunks ---


def test_get_all_chunks_returns_all_sorted_by_page(settings, ollama_client):
    docs = [
        {"id": "1", "content": "page 3", "metadata": {"filename": "doc.pdf", "page": 3}, "embedding": [1.0]},
        {"id": "2", "content": "page 1", "metadata": {"filename": "doc.pdf", "page": 1}, "embedding": [1.0]},
        {"id": "3", "content": "page 2", "metadata": {"filename": "doc.pdf", "page": 2}, "embedding": [1.0]},
    ]
    store = _make_store_with_docs(settings, ollama_client, docs)

    result = store.get_all_chunks("doc.pdf")

    assert [r["content"] for r in result] == ["page 1", "page 2", "page 3"]


def test_get_all_chunks_isolates_by_filename(settings, ollama_client):
    """Aucun chunk d'un autre fichier ne doit fuiter dans le résultat."""
    docs = [
        {"id": "1", "content": "a", "metadata": {"filename": "paysage.pdf", "page": 1}, "embedding": [1.0]},
        {"id": "2", "content": "b", "metadata": {"filename": "rappel_reseau.pdf", "page": 1}, "embedding": [1.0]},
    ]
    store = _make_store_with_docs(settings, ollama_client, docs)

    result = store.get_all_chunks("paysage.pdf")

    assert len(result) == 1
    assert result[0]["metadata"]["filename"] == "paysage.pdf"


def test_get_all_chunks_accepts_list_of_filenames(settings, ollama_client):
    docs = [
        {"id": "1", "content": "a", "metadata": {"filename": "first.pdf", "page": 1}, "embedding": [1.0]},
        {"id": "2", "content": "b", "metadata": {"filename": "second.pdf", "page": 1}, "embedding": [1.0]},
        {"id": "3", "content": "c", "metadata": {"filename": "third.pdf", "page": 1}, "embedding": [1.0]},
    ]
    store = _make_store_with_docs(settings, ollama_client, docs)

    result = store.get_all_chunks(["first.pdf", "second.pdf"])

    filenames = {r["metadata"]["filename"] for r in result}
    assert filenames == {"first.pdf", "second.pdf"}


def test_get_all_chunks_empty_filter_returns_nothing(settings, ollama_client):
    """Un filtre vide ne doit jamais être interprété comme 'tous les fichiers'."""
    docs = [
        {"id": "1", "content": "a", "metadata": {"filename": "doc.pdf", "page": 1}, "embedding": [1.0]},
    ]
    store = _make_store_with_docs(settings, ollama_client, docs)

    assert store.get_all_chunks([]) == []


# --- Tests count_pages ---


def test_count_pages_counts_distinct_pages(settings, ollama_client):
    docs = [
        {"id": "1", "content": "a", "metadata": {"filename": "doc.pdf", "page": 1}, "embedding": [1.0]},
        {"id": "2", "content": "b", "metadata": {"filename": "doc.pdf", "page": 2}, "embedding": [1.0]},
        {"id": "3", "content": "c", "metadata": {"filename": "doc.pdf", "page": 2}, "embedding": [1.0]},
    ]
    store = _make_store_with_docs(settings, ollama_client, docs)

    assert store.count_pages("doc.pdf") == 2


def test_count_pages_unknown_file_returns_zero(settings, ollama_client):
    store = _make_store_with_docs(settings, ollama_client, [])
    assert store.count_pages("inconnu.pdf") == 0
