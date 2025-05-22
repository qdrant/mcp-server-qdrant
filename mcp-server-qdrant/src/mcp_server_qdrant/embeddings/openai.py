import logging
from typing import Dict, List

import httpx

from mcp_server_qdrant.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI implementation of the embedding provider.
    
    This provider uses the OpenAI API to generate embeddings for documents and queries.
    It supports OpenAI's embedding models including text-embedding-3-small, 
    text-embedding-3-large, and text-embedding-ada-002.
    
    Args:
        model_name: The name of the OpenAI embedding model to use (e.g., 'text-embedding-3-small').
        api_key: The OpenAI API key to use for authentication.
        
    Raises:
        ValueError: If the specified model is not supported.
    """

    # Supported OpenAI models and their dimensions
    SUPPORTED_MODELS: Dict[str, int] = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, model_name: str, api_key: str) -> None:
        if not api_key:
            raise ValueError("OpenAI API key cannot be empty")
        
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model: {model_name}. "
                f"Supported models: {', '.join(self.SUPPORTED_MODELS.keys())}"
            )
        
        self.model_name = model_name
        self.api_key = api_key

    async def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Embed a list of documents into vectors using OpenAI's embedding API.
        
        Args:
            documents: List of text documents to embed.
            
        Returns:
            List of embedding vectors, one for each input document.
            
        Raises:
            httpx.HTTPError: If the API request fails.
            ValueError: If the response format is unexpected.
        """
        if not documents:
            return []
            
        try:
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
                
        except httpx.HTTPError as e:
            logger.error(f"OpenAI API request failed: {e}")
            raise
        except (KeyError, TypeError) as e:
            logger.error(f"Unexpected response format from OpenAI API: {e}")
            raise ValueError(f"Invalid response format from OpenAI API: {e}") from e

    async def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query into a vector.
        
        Args:
            query: The text query to embed.
            
        Returns:
            A single embedding vector for the query.
        """
        embeddings = await self.embed_documents([query])
        return embeddings[0]

    def get_vector_name(self) -> str:
        """
        Return the name of the vector for the Qdrant collection.
        
        For OpenAI models, this returns the model name directly to maintain
        compatibility with existing collections.
        
        Returns:
            The model name to use as the vector name.
        """
        return self.model_name

    def get_vector_size(self) -> int:
        """
        Get the size of the vector for the Qdrant collection.
        
        Returns:
            The dimension size of the embedding vectors for this model.
        """
        return self.SUPPORTED_MODELS[self.model_name]