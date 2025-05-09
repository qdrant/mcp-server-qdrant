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
    elif settings.provider_type == EmbeddingProviderType.OPENAI:
        from mcp_server_qdrant.embeddings.openai import OpenAIEmbeddingProvider

        if not settings.api_key:
            raise ValueError("API key is required for OpenAI embedding provider. Set the OPENAI_API_KEY environment variable.")
        return OpenAIEmbeddingProvider(settings.model_name, settings.api_key)
    else:
        raise ValueError(f"Unsupported embedding provider: {settings.provider_type}. Supported providers: {[t.value for t in EmbeddingProviderType]}")
