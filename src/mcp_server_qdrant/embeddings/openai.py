from openai import AsyncOpenAI, OpenAI

from mcp_server_qdrant.embeddings.base import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI-compatible embedding provider.

    Works with OpenAI directly or any compatible endpoint such as
    OpenRouter, Azure OpenAI, Together AI, and so on.

    :param model_name:  The embedding model to use (e.g. "text-embedding-3-small",
                        "qwen/qwen3-embedding-0.6b" on OpenRouter).
    :param api_key:     API key for the embedding service.
    :param base_url:    Base URL of the OpenAI-compatible endpoint.
                        Defaults to the official OpenAI API.
    :param vector_size: Dimensionality of the embedding vectors.
                        When None (default), the provider makes one synchronous
                        probe request at startup to discover the actual size.
                        Set this explicitly (e.g. 1024) to skip that probe.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        vector_size: int | None = None,
    ):
        self.model_name = model_name
        self._base_url = base_url

        self._async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        if vector_size is not None:
            self._vector_size = vector_size
        else:
            # Probe once synchronously at startup to discover the vector size.
            # This avoids maintaining a hard-coded lookup table and works with
            # any model on any compatible endpoint.
            sync_client = OpenAI(api_key=api_key, base_url=base_url)
            response = sync_client.embeddings.create(
                model=model_name,
                input=["probe"],
            )
            self._vector_size = len(response.data[0].embedding)

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Embed a list of documents into vectors."""
        response = await self._async_client.embeddings.create(
            model=self.model_name,
            input=documents,
        )
        # The API returns items sorted by index, but sort explicitly to be safe.
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]

    async def embed_query(self, query: str) -> list[float]:
        """Embed a query into a vector."""
        response = await self._async_client.embeddings.create(
            model=self.model_name,
            input=[query],
        )
        return response.data[0].embedding

    def get_vector_name(self) -> str:
        """
        Return the name of the vector for the Qdrant collection.
        Uses the last path segment of the model name, prefixed with 'openai-',
        consistent with FastEmbed's 'fast-<model>' naming convention.
        """
        slug = self.model_name.split("/")[-1].lower()
        return f"openai-{slug}"

    def get_vector_size(self) -> int:
        """Get the size of the vector for the Qdrant collection."""
        return self._vector_size
