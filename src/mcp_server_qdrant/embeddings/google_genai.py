import asyncio
from typing import List, Optional

from mcp_server_qdrant.embeddings.base import EmbeddingProvider


class GoogleGenAIProvider(EmbeddingProvider):
    """
    Google Generative AI implementation of the embedding provider.
    :param model_name: The name of the Google GenAI model to use (e.g., "text-embedding-004").
    :param api_key: Google AI API key for authentication.
    :param project: Optional Google Cloud project ID for Vertex AI.
    :param location: Optional Google Cloud location for Vertex AI.
    :param use_vertex_ai: Whether to use Vertex AI instead of Gemini Developer API.
    """

    def __init__(
        self,
        model_name: str = "text-embedding-004",
        api_key: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        use_vertex_ai: bool = False,
    ):
        self.model_name = model_name
        self.use_vertex_ai = use_vertex_ai
        self.api_key = api_key
        self.project = project
        self.location = location
        
        # Defer client initialization to when it's needed
        self._client = None

    @property
    def client(self):
        """Lazy initialization of the Google GenAI client."""
        if self._client is None:
            try:
                from google import genai
                
                # Initialize the client based on the configuration
                if self.use_vertex_ai and self.project and self.location:
                    self._client = genai.Client(
                        vertexai=True,
                        project=self.project,
                        location=self.location
                    )
                elif self.api_key:
                    self._client = genai.Client(api_key=self.api_key)
                else:
                    # Try to use environment variables
                    self._client = genai.Client()
            except ImportError:
                raise ImportError(
                    "google-genai package is required for GoogleGenAIProvider. "
                    "Install it with: pip install google-genai"
                )
        return self._client

    async def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed a list of documents into vectors."""
        # Run in a thread pool since Google GenAI client may be synchronous
        loop = asyncio.get_event_loop()
        
        def _embed_documents():
            from google.genai import types
            
            embeddings = []
            for document in documents:
                response = self.client.models.embed_content(
                    model=self.model_name,
                    contents=document,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                        output_dimensionality=None  # Use model default
                    )
                )
                embeddings.append(response.embeddings[0].values)
            return embeddings
        
        return await loop.run_in_executor(None, _embed_documents)

    async def embed_query(self, query: str) -> List[float]:
        """Embed a query into a vector."""
        # Run in a thread pool since Google GenAI client may be synchronous
        loop = asyncio.get_event_loop()
        
        def _embed_query():
            from google.genai import types
            
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=query,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY",
                    output_dimensionality=None  # Use model default
                )
            )
            return response.embeddings[0].values
        
        return await loop.run_in_executor(None, _embed_query)

    def get_vector_name(self) -> str:
        """
        Return the name of the vector for the Qdrant collection.
        """
        # Use a consistent naming convention
        model_suffix = self.model_name.replace("text-embedding-", "").replace("-", "_")
        return f"google-genai-{model_suffix}"

    def get_vector_size(self) -> int:
        """Get the size of the vector for the Qdrant collection."""
        # Known dimensions for Google GenAI embedding models
        model_dimensions = {
            "text-embedding-004": 768,
            "text-embedding-005": 768,
            "text-multilingual-embedding-002": 768,
            "gemini-embedding-001": 3072,
        }
        
        return model_dimensions.get(self.model_name, 768)  # Default to 768 