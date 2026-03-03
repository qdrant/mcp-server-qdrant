"""Google Gemini embedding provider for mcp-server-qdrant."""

from typing import Optional

from google import genai
from google.genai import types

from mcp_server_qdrant.embeddings.base import EmbeddingProvider


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using Google's Gemini API."""

    def __init__(self, model_name: str, api_key: str):
        """
        Initialize the Gemini embedding provider.

        :param model_name: The name of the Gemini model to use.
        :param api_key: The Google API key for authentication.
        """
        self._model_name = model_name.replace("models/", "")
        self._client = genai.Client(api_key=api_key)
        self._dimension: Optional[int] = None

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """
        Embed a list of documents into vectors.

        :param documents: A list of text documents to embed.
        :return: A list of embedding vectors.
        """
        response = await self._client.aio.models.embed_content(
            model=self._model_name,
            contents=documents,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
            ),
        )

        embeddings = [e.values for e in response.embeddings]

        # Cache dimension on first call
        if self._dimension is None and embeddings:
            self._dimension = len(embeddings[0])

        return embeddings

    async def embed_query(self, query: str) -> list[float]:
        """
        Embed a query into a vector.

        :param query: The query text to embed.
        :return: The embedding vector for the query.
        """
        response = await self._client.aio.models.embed_content(
            model=self._model_name,
            contents=[query],
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
            ),
        )

        embeddings = [e.values for e in response.embeddings]

        if self._dimension is None and embeddings:
            self._dimension = len(embeddings[0])

        return embeddings[0]

    def get_vector_name(self) -> str:
        """
        Get the name of the vector for the Qdrant collection.

        :return: The vector name in '{slug}' format.
        """
        slug = self._model_name.split("/")[-1]
        return f"{slug}"

    def get_vector_size(self) -> int:
        """
        Get the size of the vector for the Qdrant collection.

        :return: The dimension of the embeddings.
        """
        if self._dimension is not None:
            return self._dimension

        # Sync probe call
        response = self._client.models.embed_content(
            model=self._model_name,
            contents=["test"],
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
            ),
        )

        if response.embeddings:
            self._dimension = len(response.embeddings[0].values)

        if self._dimension is None:
            raise ValueError("Could not determine embedding dimension from Gemini API")

        return self._dimension
