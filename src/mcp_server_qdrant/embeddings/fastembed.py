import asyncio
import logging

from fastembed import TextEmbedding
from fastembed.common.model_description import DenseModelDescription, PoolingType, ModelSource

from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.settings import CustomModelSettings

logger = logging.getLogger(__name__)


class FastEmbedProvider(EmbeddingProvider):
    """
    FastEmbed implementation of the embedding provider.
    :param model_name: The name of the FastEmbed model to use.
    :param custom_model_settings: Optional custom model configuration.
    """

    def __init__(self, model_name: str, custom_model_settings: CustomModelSettings | None = None):
        self.model_name = model_name
        self.custom_model_settings = custom_model_settings
        self._cache_dir = None
        
        # Register custom model if provided
        if custom_model_settings:
            self._register_custom_model(custom_model_settings)
            self.model_name = custom_model_settings.model_name
            logger.info(f"Registered custom model: {self.model_name}")
            self._cache_dir = custom_model_settings.cache_dir
        
        self.embedding_model = TextEmbedding(
            model_name=self.model_name,
            cache_dir=self._cache_dir
        )

    def _register_custom_model(self, settings: CustomModelSettings) -> None:
        """Register a custom model with FastEmbed."""
        try:
            # Convert pooling type string to PoolingType enum
            pooling_type = getattr(PoolingType, settings.pooling_type.upper())
            
            # Create model source
            if settings.hf_model_id:
                model_source = ModelSource(hf=settings.hf_model_id)
            elif settings.model_url:
                model_source = ModelSource(url=settings.model_url)
            else:
                raise ValueError("Either hf_model_id or model_url must be provided")
            
            # Register the custom model
            TextEmbedding.add_custom_model(
                model=settings.model_name,
                pooling=pooling_type,
                normalization=settings.normalization,
                sources=model_source,
                dim=settings.vector_dimension,
                model_file=settings.model_file,
                additional_files=settings.additional_files,
            )
            
            logger.info(f"Successfully registered custom model '{settings.model_name}'")
            
        except Exception as e:
            logger.error(f"Failed to register custom model '{settings.model_name}': {e}")
            raise

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Embed a list of documents into vectors."""
        # Run in a thread pool since FastEmbed is synchronous
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: list(self.embedding_model.passage_embed(documents))
        )
        return [embedding.tolist() for embedding in embeddings]

    async def embed_query(self, query: str) -> list[float]:
        """Embed a query into a vector."""
        # Run in a thread pool since FastEmbed is synchronous
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: list(self.embedding_model.query_embed([query]))
        )
        return embeddings[0].tolist()

    def get_vector_name(self) -> str:
        """
        Return the name of the vector for the Qdrant collection.
        Important: This is compatible with the FastEmbed logic used before 0.6.0.
        """
        model_name = self.embedding_model.model_name.split("/")[-1].lower()
        return f"fast-{model_name}"

    def get_vector_size(self) -> int:
        """Get the size of the vector for the Qdrant collection."""
        # For custom models, return the configured dimension
        if self.custom_model_settings:
            return self.custom_model_settings.vector_dimension
        
        # For standard models, use the model description
        model_description: DenseModelDescription = (
            self.embedding_model._get_model_description(self.model_name)
        )
        return model_description.dim
