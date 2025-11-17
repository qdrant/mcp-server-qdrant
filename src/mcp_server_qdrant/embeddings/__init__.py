"""
Embeddings package with lazy imports for different embedding providers.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider
    from mcp_server_qdrant.embeddings.model2vec import Model2VecProvider

from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.embeddings.factory import create_embedding_provider
from mcp_server_qdrant.embeddings.types import EmbeddingProviderType

__all__ = [
    "EmbeddingProvider",
    "EmbeddingProviderType",
    "create_embedding_provider",
    "FastEmbedProvider",
    "Model2VecProvider",
]
