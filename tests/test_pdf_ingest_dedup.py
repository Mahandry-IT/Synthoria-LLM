"""Tests d'intégration pour la détection de doublons à l'upload PDF."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        app.state.ollama_client = AsyncMock()
        # Utiliser MagicMock pour le vector store (méthodes sync)
        # et configurer les méthodes async séparément
        mock_store = MagicMock()
        mock_store.has_file = MagicMock(return_value=False)
        mock_store.list_files = MagicMock(return_value=[])
        mock_store.add_chunks = AsyncMock(return_value=0)
        mock_store.search = AsyncMock(return_value=[])
        app.state.vector_store = mock_store
        app.state.gemini_client = AsyncMock()
        yield test_client


def test_ingest_duplicate_single_file_returns_failed(client):
    """Upload d'un fichier déjà ingéré → réponse avec status='failed' dans files."""
    app.state.vector_store.has_file.return_value = True

    import io
    pdf_content = b"%PDF-1.4 fake content"
    
    res = client.post(
        "/pdf/ingest",
        files=[("files", ("document.pdf", io.BytesIO(pdf_content), "application/pdf"))],
    )

    assert res.status_code == 200
    data = res.json()
    # Un seul fichier doublon → format multi avec status "failed" dans files
    assert data["status"] == "ok"
    assert len(data["files"]) == 1
    assert data["files"][0]["status"] == "failed"
    assert data["files"][0]["filename"] == "document.pdf"
    assert data["files"][0]["message"] == "File already uploaded"
    assert data["files"][0]["chunks_added"] == 0
    assert data["files"][0]["documents_added"] == 0
    assert data["total_chunks"] == 0
    assert data["total_documents"] == 0
    app.state.vector_store.has_file.assert_called_once_with("document.pdf")


def test_ingest_duplicate_multi_file_one_duplicated(client):
    """Upload de 2 fichiers dont 1 est un doublon → réponse multi avec 1 ok + 1 failed."""
    app.state.vector_store.has_file.side_effect = lambda f: f == "existing.pdf"
    app.state.vector_store.add_chunks.return_value = 10
    
    with patch("app.api.routes.extract_pdf_chunks") as mock_extract:
        mock_extract.return_value = [
            {"id": f"new.pdf::{i}::0", "content": f"chunk {i}", "metadata": {"filename": "new.pdf", "page": i+1}}
            for i in range(10)
        ]
        
        import io
        pdf_content = b"%PDF-1.4 fake content"
        
        res = client.post(
            "/pdf/ingest",
            files=[
                ("files", ("existing.pdf", io.BytesIO(pdf_content), "application/pdf")),
                ("files", ("new.pdf", io.BytesIO(pdf_content), "application/pdf")),
            ],
        )

    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert len(data["files"]) == 2
    
    # Premier fichier : doublon
    assert data["files"][0]["status"] == "failed"
    assert data["files"][0]["filename"] == "existing.pdf"
    assert data["files"][0]["message"] == "File already uploaded"
    
    # Second fichier : succès
    assert data["files"][1]["status"] == "ok"
    assert data["files"][1]["filename"] == "new.pdf"
    assert data["files"][1]["chunks_added"] == 10
    
    # Totaux
    assert data["total_chunks"] == 10
    assert data["total_documents"] == 10


def test_ingest_new_file_no_duplicate(client):
    """Upload d'un fichier nouveau → fonctionne normalement (régression)."""
    app.state.vector_store.has_file.return_value = False
    
    with patch("app.api.routes.extract_pdf_chunks") as mock_extract:
        mock_extract.return_value = [
            {"id": "test.pdf::0::0", "content": "content", "metadata": {"filename": "test.pdf", "page": 1}}
        ]
        app.state.vector_store.add_chunks.return_value = 1
        
        import io
        pdf_content = b"%PDF-1.4 fake content"
        
        res = client.post(
            "/pdf/ingest",
            files=[("files", ("test.pdf", io.BytesIO(pdf_content), "application/pdf"))],
        )

    assert res.status_code == 200
    data = res.json()
    # Un seul fichier OK → format simple (rétrocompatibilité)
    assert data["status"] == "ok"
    assert data["filename"] == "test.pdf"
    assert data["chunks_added"] == 1
    assert data["documents_added"] == 1
    app.state.vector_store.has_file.assert_called_once_with("test.pdf")
    app.state.vector_store.add_chunks.assert_awaited_once()


def test_list_files_endpoint(client):
    """GET /pdf/files retourne la liste paginée des fichiers."""
    app.state.vector_store.list_files.return_value = [
        {"id": 1, "filename": "doc1.pdf"},
        {"id": 2, "filename": "doc2.pdf"},
    ]

    res = client.get("/pdf/files")

    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert len(data["data"]) == 2
    assert data["data"][0]["id"] == 1
    assert data["data"][0]["filename"] == "doc1.pdf"
    assert data["data"][1]["id"] == 2
    assert data["data"][1]["filename"] == "doc2.pdf"
    assert data["meta"]["page"] == 1
    assert data["meta"]["total"] == 2
    assert data["meta"]["totalPages"] == 1


def test_list_files_empty(client):
    """GET /pdf/files avec un store vide → retourne liste vide."""
    app.state.vector_store.list_files.return_value = []

    res = client.get("/pdf/files")

    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["data"] == []
    assert data["meta"]["total"] == 0
    assert data["meta"]["totalPages"] == 1
