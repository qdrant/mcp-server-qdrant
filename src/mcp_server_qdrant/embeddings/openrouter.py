import logging
import os
from typing import List, Optional

import httpx
from pydantic import BaseModel

from mcp_server_qdrant.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class OpenRouterEmbeddingSettings(BaseModel):
    """Settings for OpenRouter embedding provider."""

    api_key: Optional[str] = None
    model_name: str = "text-embedding-3-small"
    base_url: str = "https://openrouter.ai/api/v1"


class OpenRouterEmbeddingProvider(EmbeddingProvider):
    """
    OpenRouter implementation of the embedding provider.
    Uses OpenRouter's API to generate embeddings for text.
    """

    def __init__(self, settings: OpenRouterEmbeddingSettings):
        self.settings = settings
        self.api_key = settings.api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable or provide in settings."
            )
        self.model_name = settings.model_name

    async def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed a list of documents into vectors."""
        if not documents:
            return []
        return await self._make_embedding_request(documents)

    async def embed_query(self, query: str) -> List[float]:
        """Embed a query into a vector."""
        if not query:
            return []
        embeddings = await self._make_embedding_request([query])
        return embeddings[0]

    async def _make_embedding_request(self, texts: List[str]) -> List[List[float]]:
        """Make the async HTTP request to OpenRouter API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "input": texts,
            "encoding_format": "float",
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.settings.base_url}/embeddings",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                return [item["embedding"] for item in data["data"]]
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"OpenRouter API error: {e.response.status_code} - {e.response.text}"
            )
        except Exception as e:
            raise RuntimeError(f"Error making OpenRouter embedding request: {str(e)}")

    def get_vector_name(self) -> str:
        """Return the name of the vector for the Qdrant collection."""
        model_name = self.model_name.replace("/", "-").replace(":", "-")
        return f"openrouter-{model_name}"

    def get_vector_size(self) -> int:
        """Get the size of the vector for the Qdrant collection."""
        env_size = os.getenv("OPENROUTER_EMBEDDING_SIZE")
        if env_size:
            return int(env_size)

        known_sizes = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
            "mistral-embed": 1024,
            "nomic-embed-text": 768,
            "nomic-ai/nomic-embed-text-v1.5": 768,
            "qwen/qwen3-embedding": 2048,
            "qwen/qwen3-embedding-8b": 4096,
        }
        if self.model_name in known_sizes:
            return known_sizes[self.model_name]

        if "large" in self.model_name:
            fallback = 3072
        else:
            fallback = 1536
        logger.warning(
            "Unknown OpenRouter model '%s': vector size defaulting to %d. "
            "Set OPENROUTER_EMBEDDING_SIZE to override.",
            self.model_name,
            fallback,
        )
        return fallback
