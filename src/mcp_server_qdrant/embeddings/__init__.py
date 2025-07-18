"""Embedding providers for the MCP server Qdrant integration."""

from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.embeddings.factory import create_embedding_provider
from mcp_server_qdrant.embeddings.types import EmbeddingProviderType

# Lazy imports to avoid dependency issues
def __getattr__(name):
    if name == "FastEmbedProvider":
        from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider
        return FastEmbedProvider
    elif name == "Model2VecProvider":
        from mcp_server_qdrant.embeddings.model2vec import Model2VecProvider
        return Model2VecProvider
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "EmbeddingProvider",
    "EmbeddingProviderType",
    "create_embedding_provider",
    "FastEmbedProvider",
    "Model2VecProvider",
]