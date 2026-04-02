from mcp_server_qdrant.embeddings.base import EmbeddingProvider


class UnnamedVectorProvider(EmbeddingProvider):
    """
    A wrapper for any embedding provider that uses unnamed vectors in Qdrant.
    This is useful when your Qdrant collections use the default vector field
    instead of named vector fields.
    """

    def __init__(self, base_provider: EmbeddingProvider):
        self.base_provider = base_provider

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Embed a list of documents into vectors."""
        return await self.base_provider.embed_documents(documents)

    async def embed_query(self, query: str) -> list[float]:
        """Embed a query into a vector."""
        return await self.base_provider.embed_query(query)

    def get_vector_name(self) -> str:
        """
        Return None to indicate unnamed vectors should be used.
        This will be handled specially in the QdrantConnector.
        """
        return "unnamed_vector"

    def get_vector_size(self) -> int:
        """Get the size of the vector for the Qdrant collection."""
        return self.base_provider.get_vector_size()