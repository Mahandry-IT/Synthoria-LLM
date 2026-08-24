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

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filename_filter: str | list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Recherche par similarité cosinus, avec filtre optionnel sur filename.

        Le filtre est appliqué AVANT le classement/troncature top_k (pré-filtrage),
        et non après : filtrer après troncature ferait perdre des chunks pertinents
        d'un fichier si un autre fichier domine le classement global.
        """
        if not self._documents:
            return []

        candidates = self._documents
        if filename_filter:
            allowed = (
                {filename_filter} if isinstance(filename_filter, str) else set(filename_filter)
            )
            candidates = [
                r for r in self._documents if r.get("metadata", {}).get("filename") in allowed
            ]
            if not candidates:
                return []

        query_embedding = await self._ollama_client.embed(query)
        scored: list[tuple[float, dict[str, Any]]] = []

        for record in candidates:
            score = self._cosine_similarity(query_embedding, record["embedding"])
            scored.append((score, record))

        scored.sort(key=lambda item: item[0], reverse=True)
        results: list[dict[str, Any]] = []
        for score, record in scored[:top_k]:
            similarity = max(min(score, 1.0), -1.0)
            results.append({
                "content": record["content"],
                "metadata": record["metadata"],
                "distance": 1.0 - similarity,
            })
        return results

    def get_all_chunks(self, filename_filter: str | list[str]) -> list[dict[str, Any]]:
        """Retourne TOUS les chunks d'un ou plusieurs fichiers, triés par page.

        Contrairement à `search`, aucune similarité cosinus n'est appliquée :
        utilisé par le mode `full_document` où l'exhaustivité prime sur la
        pertinence sémantique. Le filtrage par fichier reste strict — un
        filtre vide/None n'est jamais interprété comme "tous les fichiers"
        pour éviter toute fuite de contenu entre documents.
        """
        allowed = {filename_filter} if isinstance(filename_filter, str) else set(filename_filter)
        if not allowed:
            return []

        matching = [
            r for r in self._documents if r.get("metadata", {}).get("filename") in allowed
        ]
        matching.sort(key=lambda r: r.get("metadata", {}).get("page") or 0)

        return [
            {"content": r["content"], "metadata": r["metadata"], "distance": 0.0}
            for r in matching
        ]

    def count_pages(self, filename: str) -> int:
        """Retourne le nombre de pages distinctes indexées pour un fichier."""
        pages = {
            r.get("metadata", {}).get("page")
            for r in self._documents
            if r.get("metadata", {}).get("filename") == filename and r.get("metadata", {}).get("page") is not None
        }
        return len(pages)

    def list_files(self) -> list[dict[str, Any]]:
        """Retourne la liste des fichiers uniques dans le store avec un ID séquentiel.

        L'ID est basé sur l'ordre d'apparition dans les documents (premier fichier = id 1).
        """
        seen: dict[str, int] = {}
        files: list[dict[str, Any]] = []
        next_id = 1
        for doc in self._documents:
            filename = doc.get('metadata', {}).get('filename')
            if filename and filename not in seen:
                seen[filename] = next_id
                files.append({'id': next_id, 'filename': filename})
                next_id += 1
        return files

    def has_file(self, filename: str) -> bool:
        """Vérifie si un fichier avec le nom donné existe dans le store."""
        return any(
            doc.get('metadata', {}).get('filename') == filename
            for doc in self._documents
        )
