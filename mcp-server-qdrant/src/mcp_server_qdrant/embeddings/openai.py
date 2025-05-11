import asyncio
from typing import List
import httpx

from mcp_server_qdrant.embeddings.base import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI implementation of the embedding provider.
    :param model_name: The name of the OpenAI embedding model to use (e.g., 'text-embedding-3-small').
    :param api_key: The OpenAI API key to use for authentication.
    """

    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key
        # Mapping of model names to their dimensions
        self.dimension_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        if model_name not in self.dimension_map:
            raise ValueError(f"Unsupported model: {model_name}. Supported models: {', '.join(self.dimension_map.keys())}")

    async def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed a list of documents into vectors."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={"input": documents, "model": self.model_name},
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
            # Sort by index to ensure the order matches the input
            embeddings_data = sorted(data["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in embeddings_data]

    async def embed_query(self, query: str) -> List[float]:
        """Embed a query into a vector."""
        embeddings = await self.embed_documents([query])
        return embeddings[0]

    def get_vector_name(self) -> str:
        """
        Return the name of the vector for the Qdrant collection.
        """
        return 

    def get_vector_size(self) -> int:
        """Get the size of the vector for the Qdrant collection."""
        return self.dimension_map[self.model_name]