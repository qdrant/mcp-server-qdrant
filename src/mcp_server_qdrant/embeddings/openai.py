"""OpenAI-compatible embedding provider for mcp-server-qdrant.

Enables using any OpenAI-compatible embedding API (e.g. local inference
servers, Azure OpenAI, LiteLLM, Ollama, etc.) as embedding backend.

Configuration via environment variables:
    EMBEDDING_PROVIDER=openai
    EMBEDDING_MODEL=<model-name>
    OPENAI_BASE_URL=http://your-server:8090/v1
    OPENAI_API_KEY=<optional, defaults to "not-needed">

Uses only Python stdlib (urllib) — no additional dependencies required.
"""

import asyncio
import json
import urllib.request
from typing import Optional

from mcp_server_qdrant.embeddings.base import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using an OpenAI-compatible HTTP API.

    This provider calls a ``/v1/embeddings`` endpoint that follows the
    `OpenAI Embeddings API specification
    <https://platform.openai.com/docs/api-reference/embeddings>`_.

    Only Python standard-library networking (``urllib``) is used so that
    no extra dependencies are pulled in.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "not-needed",
    ):
        self._model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._dimension: Optional[int] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        """Send an embedding request and return the resulting vectors."""
        url = f"{self._base_url}/embeddings"
        payload = json.dumps(
            {"input": texts, "model": self._model_name}
        ).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        embeddings = [item["embedding"] for item in data["data"]]

        if self._dimension is None and embeddings:
            self._dimension = len(embeddings[0])

        return embeddings

    # ------------------------------------------------------------------
    # EmbeddingProvider interface
    # ------------------------------------------------------------------

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._call_api, documents)

    async def embed_query(self, query: str) -> list[float]:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._call_api, [query])
        return result[0]

    def get_vector_name(self) -> str:
        # Derive a short slug from the model name, e.g.
        # "nomic-ai/nomic-embed-text-v1.5" -> "openai-nomic-embed-text-v1.5"
        slug = self._model_name.split("/")[-1].lower()
        return f"openai-{slug}"

    def get_vector_size(self) -> int:
        if self._dimension is not None:
            return self._dimension
        # Probe the API with a short text to discover the vector size.
        self._call_api(["dimension probe"])
        assert self._dimension is not None, "Failed to determine embedding dimension"
        return self._dimension
