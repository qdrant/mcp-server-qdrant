from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.embeddings.types import EmbeddingProviderType
from mcp_server_qdrant.settings import EmbeddingProviderSettings


def create_embedding_provider(settings: EmbeddingProviderSettings) -> EmbeddingProvider:
    """
    Create an embedding provider based on the specified type.
    :param settings: The settings for the embedding provider.
    :return: An instance of the specified embedding provider.
    """
    if settings.provider_type == EmbeddingProviderType.FASTEMBED:
        from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider

        provider = FastEmbedProvider(settings.model_name)
        
        # Wrap with UnnamedVectorProvider if requested
        if settings.use_unnamed_vectors:
            from mcp_server_qdrant.embeddings.unnamed import UnnamedVectorProvider
            provider = UnnamedVectorProvider(provider)
        
        return provider
    elif settings.provider_type == EmbeddingProviderType.FASTEMBED_UNNAMED:
        from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider
        from mcp_server_qdrant.embeddings.unnamed import UnnamedVectorProvider

        base_provider = FastEmbedProvider(settings.model_name)
        return UnnamedVectorProvider(base_provider)
    else:
        raise ValueError(f"Unsupported embedding provider: {settings.provider_type}")
