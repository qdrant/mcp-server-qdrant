"""
Unnamed vector provider wrapper.

This provider wraps another embedding provider and returns "unnamed_vector"
as the vector name, which allows using Qdrant's default unnamed vector field.
"""

from mcp_server_qdrant.embeddings.base import EmbeddingProvider


class UnnamedVectorProvider(EmbeddingProvider):
    """
    Wrapper for embedding providers to use unnamed vectors in Qdrant.
    """

    def __init__(self, base_provider: EmbeddingProvider):
        """
        Initialize the unnamed vector provider wrapper.
        :param base_provider: The base embedding provider to wrap
        """
        self.base_provider = base_provider

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """
        Embed multiple documents.
        :param documents: List of text documents to embed
        :return: List of embedding vectors
        """
        return await self.base_provider.embed_documents(documents)

    async def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query.
        :param query: Query text to embed
        :return: Embedding vector
        """
        return await self.base_provider.embed_query(query)

    def get_vector_name(self) -> str:
        """
        Get the name identifier for this vector in Qdrant.
        Returns "unnamed_vector" to use Qdrant's default unnamed vector.
        :return: "unnamed_vector"
        """
        return "unnamed_vector"

    def get_vector_size(self) -> int:
        """
        Get the dimension of the embedding vectors.
        :return: Vector dimension from base provider
        """
        return self.base_provider.get_vector_size()
