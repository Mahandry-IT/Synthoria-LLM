import os
from typing import Any

import chromadb

from app.services.ollama_client import OllamaClient
from app.core.config import Settings


class ChromaVectorStore:
    def __init__(self, settings: Settings, ollama_client: OllamaClient) -> None:
        self._settings = settings
        self._ollama_client = ollama_client
        self._client = chromadb.PersistentClient(path=settings.chroma_persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    async def add_chunks(self, chunks: list[dict[str, Any]]) -> int:
        if not chunks:
            return 0

        ids: list[str] = []
        docs: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for chunk in chunks:
            ids.append(chunk["id"])
            docs.append(chunk["content"])
            metadatas.append(chunk["metadata"])

        embeddings: list[list[float]] = []
        for doc in docs:
            embedding = await self._ollama_client.embed(doc)
            embeddings.append(embedding)

        self._collection.add(documents=docs, embeddings=embeddings, ids=ids, metadatas=metadatas)
        return len(chunks)

    async def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        query_vector = await self._ollama_client.embed(query)
        results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        final: list[dict[str, Any]] = []
        if not results or not results.get("documents"):
            return final

        for idx, doc in enumerate(results["documents"][0]):
            final.append(
                {
                    "content": doc,
                    "metadata": results["metadatas"][0][idx],
                    "distance": results["distances"][0][idx],
                }
            )
        return final
