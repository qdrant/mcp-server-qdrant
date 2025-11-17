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
        if settings.use_unnamed_vectors:
            from mcp_server_qdrant.embeddings.unnamed import UnnamedVectorProvider

            return UnnamedVectorProvider(provider)
        return provider
    elif settings.provider_type == EmbeddingProviderType.FASTEMBED_UNNAMED:
        from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider
        from mcp_server_qdrant.embeddings.unnamed import UnnamedVectorProvider

        provider = FastEmbedProvider(settings.model_name)
        return UnnamedVectorProvider(provider)
    elif settings.provider_type == EmbeddingProviderType.MODEL2VEC:
        from mcp_server_qdrant.embeddings.model2vec import Model2VecProvider

        return Model2VecProvider(settings.model_name)
    elif settings.provider_type == EmbeddingProviderType.OAI_COMPAT:
        from mcp_server_qdrant.embeddings.oai_compat import OAICompatProvider

        return OAICompatProvider(
            model_name=settings.model_name,
            endpoint_url=settings.oai_compat_endpoint,
            api_key=settings.oai_compat_api_key,
            vector_size=settings.oai_compat_vec_size,
        )
    else:
        raise ValueError(f"Unsupported embedding provider: {settings.provider_type}")
