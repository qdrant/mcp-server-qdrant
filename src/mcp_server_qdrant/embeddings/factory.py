from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.embeddings.types import EmbeddingProviderType


def create_embedding_provider(provider_type: str, model_name: str) -> EmbeddingProvider:
    """
    Create an embedding provider based on the specified type.

    :param provider_type: The type of embedding provider to create.
    :param model_name: The name of the model to use for embeddings, specific to the provider type.
    :return: An instance of the specified embedding provider.
    """
    # Normalize the provider type to handle different formats
    provider_type_lower = provider_type.lower()
    
    if provider_type_lower == EmbeddingProviderType.FASTEMBED:
        from .fastembed import FastEmbedProvider
        return FastEmbedProvider(model_name)
    else:
        raise ValueError(f"Unsupported embedding provider: {provider_type}")
