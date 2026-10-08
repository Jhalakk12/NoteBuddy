"""Persistent Chroma storage for course-material chunks."""

import os
from pathlib import Path
from typing import Protocol

import chromadb

from knowledge.models import Chunk, RetrievedChunk


class Embedder(Protocol):
    model: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class KnowledgeStore:
    def __init__(
        self,
        persist_directory: Path,
        embedder: Embedder,
        collection_name: str = "course_materials",
    ) -> None:
        chroma_host = os.getenv("CHROMA_HOST", "").strip()
        if chroma_host:
            self.client = chromadb.HttpClient(
                host=chroma_host,
                port=int(os.getenv("CHROMA_PORT", "8000")),
                ssl=os.getenv("CHROMA_SSL", "false").lower() == "true",
            )
        else:
            persist_directory.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(persist_directory))
        self.embedder = embedder
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": embedder.model,
            },
        )

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception as exc:
            if "does not exist" not in str(exc).lower():
                raise
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": self.embedder.model,
            },
        )

    def upsert(self, chunks: list[Chunk], batch_size: int = 32) -> int:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            texts = [chunk.text for chunk in batch]
            self.collection.upsert(
                ids=[chunk.id for chunk in batch],
                documents=texts,
                metadatas=[chunk.metadata() for chunk in batch],
                embeddings=self.embedder.embed(texts),
            )
        return len(chunks)

    def count(self) -> int:
        return self.collection.count()

    def delete_source(self, source: str) -> None:
        """Remove only this document's vectors without re-embedding other files."""
        self.collection.delete(where={"source": source})

    def sample(self, limit: int = 5) -> dict:
        return self.collection.get(limit=limit, include=["documents", "metadatas"])

    def query(self, question: str, top_k: int = 4) -> list[RetrievedChunk]:
        if self.count() == 0:
            return []
        query_embedding = self.embedder.embed([question])[0]
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        rows: list[RetrievedChunk] = []
        for text, metadata, distance in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
            strict=True,
        ):
            stored_page = int(metadata.get("page", 0))
            rows.append(
                RetrievedChunk(
                    text=text,
                    source=str(metadata["source"]),
                    page=stored_page or None,
                    section=str(metadata["section"]),
                    citation=str(metadata["citation"]),
                    distance=float(distance),
                )
            )
        return rows
