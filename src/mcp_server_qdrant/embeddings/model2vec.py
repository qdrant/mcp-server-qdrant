import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from model2vec import StaticModel

from mcp_server_qdrant.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class Model2VecProvider(EmbeddingProvider):
    """
    Embedding provider using Model2Vec models.
    Model2Vec provides fast, lightweight static embeddings.
    """

    def __init__(self, model_name: str):
        """
        Initialize the Model2Vec provider.
        :param model_name: Name of the Model2Vec model to use
        """
        self.model_name = model_name
        logger.info(f"Loading Model2Vec model: {model_name}")
        self.model = StaticModel.from_pretrained(model_name)
        self.vector_size = self.model.dim
        self.executor = ThreadPoolExecutor(max_workers=1)

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """
        Embed multiple documents.
        :param documents: List of text documents to embed
        :return: List of embedding vectors
        """
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            self.executor, self._embed_sync, documents
        )
        return embeddings

    async def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query.
        :param query: Query text to embed
        :return: Embedding vector
        """
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            self.executor, self._embed_sync, [query]
        )
        return embeddings[0]

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        """
        Synchronous embedding method.
        :param texts: List of texts to embed
        :return: List of embedding vectors
        """
        embeddings = self.model.encode(texts)
        return [embedding.tolist() for embedding in embeddings]

    def get_vector_name(self) -> str:
        """
        Get the name identifier for this vector in Qdrant.
        :return: Vector name (e.g., "m2v-minishlab/potion-base-8M")
        """
        # Use a shortened model name for the vector identifier
        model_short_name = self.model_name.split("/")[-1] if "/" in self.model_name else self.model_name
        return f"m2v-{model_short_name}"

    def get_vector_size(self) -> int:
        """
        Get the dimension of the embedding vectors.
        :return: Vector dimension
        """
        return self.vector_size
