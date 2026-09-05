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
        from mcp_server_qdrant.embeddings.openai import OpenAIProvider

        return OpenAIProvider(
            settings.model_name,
            base_url=settings.base_url,
            api_key=settings.api_key,
            vector_size=settings.vector_size,
            vector_name=settings.vector_name,
            query_prefix=settings.query_prefix,
            document_prefix=settings.document_prefix,
        )
    else:
        raise ValueError(f"Unsupported embedding provider: {settings.provider_type}")
