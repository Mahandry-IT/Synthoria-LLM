import asyncio
import logging

import httpx

from app.core.config import Settings
from app.core.exceptions import (
    OllamaModelNotFoundError,
    OllamaUnavailableError,
)

logger = logging.getLogger(__name__)


class OllamaClient:
    """Encapsule les appels HTTP vers l'API Ollama (retry + timeout inclus)."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self._settings = settings
        self._client = client or httpx.AsyncClient(
            base_url=settings.ollama_base_url,
            timeout=settings.ollama_timeout_seconds,
        )

    async def generate(self, prompt: str, model: str, stream: bool = False) -> dict:
        """
        Envoie un prompt à Ollama et retourne la réponse générée.

        Paramètres:
            prompt: texte d'entrée.
            model: nom du modèle Ollama (ex: "llama3.2").
            stream: si False, agrège la réponse complète côté client.

        Retour: dict avec les clés "model", "response", "done".

        Lève:
            OllamaUnavailableError: connexion/timeout impossible après retries.
            OllamaModelNotFoundError: modèle absent (404 Ollama).
        """
        payload = {"model": model, "prompt": prompt, "stream": stream}
        last_error: Exception | None = None

        for attempt in range(self._settings.ollama_max_retries):
            try:
                response = await self._client.post("/api/generate", json=payload)
                if response.status_code == 404:
                    raise OllamaModelNotFoundError(f"Modèle introuvable: {model}")
                response.raise_for_status()
                return response.json()
            except OllamaModelNotFoundError:
                raise
            except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as exc:
                last_error = exc
                logger.warning(
                    "ollama_call_failed",
                    extra={"attempt": attempt + 1, "error": str(exc)},
                )
                if attempt < self._settings.ollama_max_retries - 1:
                    await asyncio.sleep(min(2**attempt, 10))

        raise OllamaUnavailableError(
            f"Ollama injoignable après {self._settings.ollama_max_retries} tentatives"
        ) from last_error

    async def embed(self, text: str, model: str | None = None) -> list[float]:
        """Retourne un vecteur d'embedding via l'API d'Ollama.

        Utilise le endpoint /api/embed (Ollama ≥ 0.9) — l'ancien
        /api/embeddings retourne un tableau vide sur les versions récentes.
        Fallback automatique si le nouveau endpoint n'est pas disponible.
        """
        payload = {"model": model or self._settings.ollama_embedding_model, "input": text}
        last_error: Exception | None = None

        for attempt in range(self._settings.ollama_max_retries):
            try:
                response = await self._client.post("/api/embed", json=payload)
                if response.status_code == 404:
                    # Fallback: ancien endpoint pour Ollama < 0.9
                    response = await self._client.post("/api/embeddings", json=payload)
                if response.status_code == 404:
                    raise OllamaModelNotFoundError(f"Modèle d'embedding introuvable: {payload['model']}")
                response.raise_for_status()
                data = response.json()

                if isinstance(data, dict):
                    # /api/embed → {"embeddings": [[...]]}
                    if "embeddings" in data and isinstance(data["embeddings"], list):
                        first = data["embeddings"][0]
                        if isinstance(first, list) and first:
                            return first
                    # Fallback ancien format → {"embedding": [...]}
                    if "embedding" in data and isinstance(data["embedding"], list) and data["embedding"]:
                        return data["embedding"]
                raise ValueError("Format de réponse embedding inattendu")
            except (OllamaModelNotFoundError, ValueError):
                raise
            except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as exc:
                last_error = exc
                logger.warning(
                    "ollama_embedding_failed",
                    extra={"attempt": attempt + 1, "error": str(exc)},
                )
                if attempt < self._settings.ollama_max_retries - 1:
                    await asyncio.sleep(min(2**attempt, 10))

        raise OllamaUnavailableError(
            f"Ollama embedding injoignable après {self._settings.ollama_max_retries} tentatives"
        ) from last_error

    async def is_reachable(self) -> bool:
        try:
            response = await self._client.get("/api/tags")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def close(self) -> None:
        await self._client.aclose()
