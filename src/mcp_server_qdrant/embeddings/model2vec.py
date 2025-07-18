import asyncio
from typing import Optional

from model2vec import StaticModel

from mcp_server_qdrant.embeddings.base import EmbeddingProvider


class Model2VecProvider(EmbeddingProvider):
    """
    Model2Vec implementation of the embedding provider.
    :param model_name: The name of the Model2Vec model to use.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model: Optional[StaticModel] = None
        self._vector_size: Optional[int] = None
        
        # Initialize the model
        try:
            self._model = StaticModel.from_pretrained(model_name)
            # Get vector size by encoding a test string
            test_embedding = self._model.encode(["test"])
            self._vector_size = len(test_embedding[0])
        except Exception as e:
            raise ValueError(f"Failed to load model2vec model '{model_name}': {str(e)}")

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Embed a list of documents into vectors."""
        # Validate input types first
        if not isinstance(documents, list):
            raise TypeError("Documents must be a list of strings")
        
        if not documents:
            return []
        
        for i, doc in enumerate(documents):
            if not isinstance(doc, str):
                raise TypeError(f"Document at index {i} must be a string, got {type(doc)}")
        
        # Run in a thread pool since Model2Vec is synchronous
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: self._model.encode(documents)
        )
        
        # Convert to list of lists and ensure consistent dimensionality
        result = [embedding.tolist() for embedding in embeddings]
        
        # Verify consistent dimensionality across all embeddings
        if result and len(set(len(emb) for emb in result)) > 1:
            raise RuntimeError("Inconsistent vector dimensions detected across documents")
        
        return result

    async def embed_query(self, query: str) -> list[float]:
        """Embed a query into a vector."""
        # Validate input type
        if not isinstance(query, str):
            raise TypeError(f"Query must be a string, got {type(query)}")
        
        # Run in a thread pool since Model2Vec is synchronous
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: self._model.encode([query])
        )
        
        # Convert to list and ensure consistent dimensionality with documents
        result = embeddings[0].tolist()
        
        # Verify the query embedding has the expected dimensionality
        if len(result) != self._vector_size:
            raise RuntimeError(f"Query embedding dimension mismatch: expected {self._vector_size}, got {len(result)}")
        
        return result

    def get_vector_name(self) -> str:
        """
        Return the name of the vector for the Qdrant collection.
        Uses the pattern "m2v-{simplified_model_name}".
        """
        # Simplify model name by taking the last part after "/" and converting to lowercase
        simplified_name = self.model_name.split("/")[-1].lower()
        return f"m2v-{simplified_name}"

    def get_vector_size(self) -> int:
        """Get the size of the vector for the Qdrant collection."""
        if self._vector_size is None:
            raise RuntimeError("Model not properly initialized")
        return self._vector_size