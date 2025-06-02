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

    elif settings.provider_type == EmbeddingProviderType.GOOGLE_GENAI:
        from mcp_server_qdrant.embeddings.google_genai import GoogleGenAIProvider

        return GoogleGenAIProvider(
            model_name=settings.model_name,
            api_key=getattr(settings, "api_key", None),
            project=getattr(settings, "project", None),
            location=getattr(settings, "location", None),
            use_vertex_ai=getattr(settings, "use_vertex_ai", False),
        )

    else:
        raise ValueError(f"Unsupported embedding provider: {settings.provider_type}")
