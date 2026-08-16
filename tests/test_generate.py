from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.core.exceptions import OllamaModelNotFoundError, OllamaUnavailableError
from app.main import app


@pytest.fixture
def client():
    mock_ollama = AsyncMock()
    with TestClient(app) as test_client:
        app.state.ollama_client = mock_ollama  # écrase le client réel créé par le lifespan
        yield test_client, mock_ollama


def test_generate_success(client):
    test_client, mock_ollama = client
    mock_ollama.generate.return_value = {"model": "llama3.2", "response": "salut", "done": True}

    res = test_client.post("/generate", json={"prompt": "bonjour"})

    assert res.status_code == 200
    assert res.json() == {"model": "llama3.2", "response": "salut", "done": True}


def test_generate_empty_prompt_rejected(client):
    test_client, _ = client
    res = test_client.post("/generate", json={"prompt": "   "})
    assert res.status_code == 422


def test_generate_model_not_found(client):
    test_client, mock_ollama = client
    mock_ollama.generate.side_effect = OllamaModelNotFoundError("nope")

    res = test_client.post("/generate", json={"prompt": "hi", "model": "inconnu"})

    assert res.status_code == 404


def test_generate_ollama_unavailable(client):
    test_client, mock_ollama = client
    mock_ollama.generate.side_effect = OllamaUnavailableError("down")

    res = test_client.post("/generate", json={"prompt": "hi"})

    assert res.status_code == 503


def test_health(client):
    test_client, mock_ollama = client
    mock_ollama.is_reachable.return_value = True

    res = test_client.get("/health")

    assert res.status_code == 200
    assert res.json() == {"status": "ok", "ollama_reachable": True}
