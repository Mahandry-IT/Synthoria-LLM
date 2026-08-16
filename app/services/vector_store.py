import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import Settings
from app.services.ollama_client import OllamaClient


class NumpyVectorStore:
    def __init__(self, settings: Settings, ollama_client: OllamaClient) -> None:
        self._settings = settings
        self._ollama_client = ollama_client
        self._persist_dir = Path(settings.chroma_persist_directory)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._persist_dir / f"{settings.chroma_collection_name}.json"
        self._documents: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if self._index_path.exists():
            try:
                self._documents = json.loads(self._index_path.read_text(encoding="utf-8"))
            except Exception:
                self._documents = []

    def _save(self) -> None:
        self._index_path.write_text(json.dumps(self._documents, ensure_ascii=False), encoding="utf-8")

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        a_arr = np.asarray(a, dtype=float)
        b_arr = np.asarray(b, dtype=float)
        if np.allclose(a_arr, 0) or np.allclose(b_arr, 0):
            return 0.0
        return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))

    async def add_chunks(self, chunks: list[dict[str, Any]]) -> int:
        if not chunks:
            return 0

        for chunk in chunks:
            embedding = await self._ollama_client.embed(chunk["content"])
            self._documents.append({
                "id": chunk["id"],
                "content": chunk["content"],
                "metadata": chunk["metadata"],
                "embedding": embedding,
            })

        self._save()
        return len(chunks)

    async def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not self._documents:
            return []

        query_embedding = await self._ollama_client.embed(query)
        scored: list[tuple[float, dict[str, Any]]] = []

        for record in self._documents:
            score = self._cosine_similarity(query_embedding, record["embedding"])
            scored.append((score, record))

        scored.sort(key=lambda item: item[0], reverse=True)
        results: list[dict[str, Any]] = []
        for _, record in scored[:top_k]:
            results.append({
                "content": record["content"],
                "metadata": record["metadata"],
                "distance": 1.0 - max(min(record["embedding"][0] if record["embedding"] else 0.0, 1.0), 0.0),
            })
        return results
