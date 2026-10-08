"""Ollama embedding client used by the ingestion and later retrieval stages."""

from typing import Any

import httpx


class EmbeddingError(RuntimeError):
    """Raised when Ollama cannot create embeddings."""


class OllamaEmbeddingClient:
    def __init__(
        self,
        base_url: str,
        model: str = "nomic-embed-text",
        timeout_seconds: float = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload: dict[str, Any] = {"model": self.model, "input": texts}
        try:
            response = httpx.post(
                f"{self.base_url}/api/embed",
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            embeddings = response.json()["embeddings"]
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise EmbeddingError(
                f"Could not create embeddings with {self.model} at {self.base_url}: {exc}"
            ) from exc

        if len(embeddings) != len(texts):
            raise EmbeddingError("Ollama returned the wrong number of embeddings.")
        return embeddings

