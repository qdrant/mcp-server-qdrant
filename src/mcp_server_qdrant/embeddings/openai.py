import os

from openai import AsyncOpenAI

from mcp_server_qdrant.embeddings.base import EmbeddingProvider

# OpenAI embedding model dimensions
MODEL_DIMENSIONS = {
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}

DEFAULT_MODEL = "text-embedding-3-small"


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI implementation of the embedding provider.
    :param model_name: The name of the OpenAI embedding model to use.
    :param api_key: Optional API key. If not provided, uses OPENAI_API_KEY env var.
    """

    def __init__(self, model_name: str | None = None, api_key: str | None = None):
        self.model_name = model_name or DEFAULT_MODEL
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self._client = AsyncOpenAI(api_key=self._api_key)

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Embed a list of documents into vectors."""
        response = await self._client.embeddings.create(
            model=self.model_name,
            input=documents,
        )
        return [item.embedding for item in response.data]

    async def embed_query(self, query: str) -> list[float]:
        """Embed a query into a vector."""
        response = await self._client.embeddings.create(
            model=self.model_name,
            input=[query],
        )
        return response.data[0].embedding

    def get_vector_name(self) -> str:
        """
        Return the name of the vector for the Qdrant collection.
        Uses a format similar to FastEmbed for consistency.
        """
        model_name = self.model_name.replace("/", "-").lower()
        return f"openai-{model_name}"

    def get_vector_size(self) -> int:
        """Get the size of the vector for the Qdrant collection."""
        if self.model_name in MODEL_DIMENSIONS:
            return MODEL_DIMENSIONS[self.model_name]
        # Default to 1536 for unknown models (most common dimension)
        return 1536
