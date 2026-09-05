import os
import re

from openai import AsyncOpenAI, OpenAI

from mcp_server_qdrant.embeddings.base import EmbeddingProvider


class OpenAIProvider(EmbeddingProvider):
    """
    OpenAI-compatible implementation of the embedding provider.
    Works with the OpenAI API and with any server exposing an `/v1/embeddings` endpoint.
    :param model_name: The name of the embedding model to use.
    :param base_url: The base URL of the API, including the `/v1` suffix.
    :param api_key: The API key to use. Falls back to `OPENAI_API_KEY`, then to a placeholder,
                    as local servers typically do not check it.
    :param vector_size: The dimensionality of the embeddings. Detected from the API if not provided.
    :param vector_name: The name of the vector in the Qdrant collection.
    :param query_prefix: A string prepended to every query before embedding it.
    :param document_prefix: A string prepended to every document before embedding it.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        vector_size: int | None = None,
        vector_name: str | None = None,
        query_prefix: str = "",
        document_prefix: str = "",
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or "unset"
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self._vector_size = vector_size
        self._vector_name = vector_name
        self._client = AsyncOpenAI(base_url=base_url, api_key=self.api_key)

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Embed a list of documents into vectors."""
        response = await self._client.embeddings.create(
            model=self.model_name,
            input=[f"{self.document_prefix}{document}" for document in documents],
        )
        return [item.embedding for item in sorted(response.data, key=lambda d: d.index)]

    async def embed_query(self, query: str) -> list[float]:
        """Embed a query into a vector."""
        response = await self._client.embeddings.create(
            model=self.model_name,
            input=[f"{self.query_prefix}{query}"],
        )
        return response.data[0].embedding

    def get_vector_name(self) -> str:
        """Return the name of the vector for the Qdrant collection."""
        if self._vector_name is None:
            model_name = self.model_name.split("/")[-1].lower()
            self._vector_name = "openai-" + re.sub(
                r"[^a-z0-9]+", "-", model_name
            ).strip("-")
        return self._vector_name

    def get_vector_size(self) -> int:
        """
        Get the size of the vector for the Qdrant collection.
        The API does not advertise it, so an unconfigured size is detected by embedding a probe document.
        """
        if self._vector_size is None:
            client = OpenAI(base_url=self.base_url, api_key=self.api_key)
            response = client.embeddings.create(model=self.model_name, input=["probe"])
            self._vector_size = len(response.data[0].embedding)
        return self._vector_size
