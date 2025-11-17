import logging

from openai import AsyncOpenAI

from mcp_server_qdrant.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class OAICompatProvider(EmbeddingProvider):
    """
    Embedding provider using OpenAI-compatible API endpoints.
    This provider works with OpenAI's API and any compatible services.
    """

    def __init__(
        self,
        model_name: str,
        endpoint_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        vector_size: int | None = None,
    ):
        """
        Initialize the OpenAI-compatible provider.
        :param model_name: Name of the embedding model to use
        :param endpoint_url: API endpoint URL (default: OpenAI's endpoint)
        :param api_key: API key for authentication
        :param vector_size: Optional override for vector dimension
        """
        self.model_name = model_name
        self.endpoint_url = endpoint_url
        self.vector_size = vector_size
        logger.info(
            f"Initializing OAI-compatible provider with model: {model_name}, endpoint: {endpoint_url}"
        )

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=endpoint_url,
        )

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """
        Embed multiple documents.
        :param documents: List of text documents to embed
        :return: List of embedding vectors
        """
        response = await self.client.embeddings.create(
            input=documents,
            model=self.model_name,
        )
        return [item.embedding for item in response.data]

    async def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query.
        :param query: Query text to embed
        :return: Embedding vector
        """
        response = await self.client.embeddings.create(
            input=[query],
            model=self.model_name,
        )
        return response.data[0].embedding

    def get_vector_name(self) -> str:
        """
        Get the name identifier for this vector in Qdrant.
        :return: Vector name based on model name
        """
        # Use model name as vector identifier
        return self.model_name.replace("/", "-").replace(":", "-")

    def get_vector_size(self) -> int:
        """
        Get the dimension of the embedding vectors.
        If not set during initialization, it will be fetched from the API.
        :return: Vector dimension
        """
        if self.vector_size is None:
            # Lazy fetch vector size if not provided
            import asyncio

            self.vector_size = asyncio.run(self.fetch_vec_size())
        return self.vector_size

    async def fetch_vec_size(self) -> int:
        """
        Fetch the vector size by making a test embedding call.
        :return: Vector dimension
        """
        logger.info(f"Fetching vector size for model: {self.model_name}")
        # Make a test call with a dummy input to determine vector size
        response = await self.client.embeddings.create(
            input=["test"],
            model=self.model_name,
        )
        vec_size = len(response.data[0].embedding)
        logger.info(f"Detected vector size: {vec_size}")
        return vec_size
