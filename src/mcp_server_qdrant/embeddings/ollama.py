"""
Ollama embedding provider for local embedding models.

Uses the Ollama API to generate embeddings locally without external API keys.
Supports any model available in Ollama (e.g., bge-m3, nomic-embed-text, mxbai-embed-large).
"""

import asyncio
from typing import Any

import httpx

from mcp_server_qdrant.embeddings.base import EmbeddingProvider


class OllamaProvider(EmbeddingProvider):
    """
    Ollama implementation of the embedding provider.
    
    :param model_name: The name of the Ollama model to use (e.g., "bge-m3").
    :param base_url: The base URL of the Ollama API (default: "http://localhost:11434").
    """

    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self._vector_size: int | None = None

    async def _embed(self, texts: list[str], input_type: str = "search_document") -> list[list[float]]:
        """Embed a list of texts using the Ollama API."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/embed",
                json={
                    "model": self.model_name,
                    "input": texts,
                },
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()
            return data["embeddings"]

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Embed a list of documents into vectors."""
        return await self._embed(documents, input_type="search_document")

    async def embed_query(self, query: str) -> list[float]:
        """Embed a query into a vector."""
        embeddings = await self._embed([query], input_type="search_query")
        return embeddings[0]

    def get_vector_name(self) -> str:
        """Return the name of the vector for the Qdrant collection."""
        model_name = self.model_name.split("/")[-1].lower()
        return f"ollama-{model_name}"

    async def get_vector_size(self) -> int:
        """Get the size of the vector for the Qdrant collection."""
        if self._vector_size is None:
            # Embed a dummy text to get the vector size
            embeddings = await self._embed(["test"])
            self._vector_size = len(embeddings[0])
        return self._vector_size
