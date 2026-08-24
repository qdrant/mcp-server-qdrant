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

        return FastEmbedProvider(settings.model_name)
    elif settings.provider_type == EmbeddingProviderType.OPENROUTER:
        from mcp_server_qdrant.embeddings.openrouter import OpenRouterEmbeddingProvider, OpenRouterEmbeddingSettings

        # Create OpenRouter settings from the general settings
        openrouter_settings = OpenRouterEmbeddingSettings(
            api_key=settings.openrouter_api_key,
            model_name=settings.model_name,
        )
        return OpenRouterEmbeddingProvider(openrouter_settings)
    else:
        raise ValueError(f"Unsupported embedding provider: {settings.provider_type}")
