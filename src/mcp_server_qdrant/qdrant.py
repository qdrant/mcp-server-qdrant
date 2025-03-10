import uuid
from typing import Any, Dict, List, Optional

from qdrant_client import AsyncQdrantClient, models

from .embeddings.base import EmbeddingProvider


class QdrantConnector:
    """
    Encapsulates the connection to a Qdrant server and all the methods to interact with it.
    :param qdrant_url: The URL of the Qdrant server.
    :param qdrant_api_key: The API key to use for the Qdrant server.
    :param collection_name: The name of the collection to use.
    :param embedding_provider: The embedding provider to use.
    :param qdrant_local_path: The path to the storage directory for the Qdrant client, if local mode is used.
    """

    def __init__(
        self,
        qdrant_url: Optional[str],
        qdrant_api_key: Optional[str],
        collection_name: str,
        embedding_provider: EmbeddingProvider,
        qdrant_local_path: Optional[str] = None,
    ):
        self._qdrant_url = qdrant_url.rstrip("/") if qdrant_url else None
        self._qdrant_api_key = qdrant_api_key
        self._collection_name = collection_name
        self._embedding_provider = embedding_provider
        self._client = AsyncQdrantClient(
            location=qdrant_url, api_key=qdrant_api_key, path=qdrant_local_path
        )

    async def _ensure_collection_exists(self):
        """Ensure that the collection exists, creating it if necessary."""
        collection_exists = await self._client.collection_exists(self._collection_name)
        if not collection_exists:
            # Create the collection with the appropriate vector size
            # We'll get the vector size by embedding a sample text
            sample_vector = await self._embedding_provider.embed_query("sample text")
            vector_size = len(sample_vector)

            # Use the vector name as defined in the embedding provider
            vector_name = self._embedding_provider.get_vector_name()
            await self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config={
                    vector_name: models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                    )
                },
            )

    async def store_memory(self, information: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Store a memory in the Qdrant collection.
        :param information: The information to store.
        :param metadata: Optional metadata to associate with the memory.
        """
        await self._ensure_collection_exists()

        # Embed the document
        embeddings = await self._embedding_provider.embed_documents([information])

        # Prepare payload with document and optional metadata
        payload = {"document": information}
        if metadata:
            payload["metadata"] = metadata
            
        # Add to Qdrant
        vector_name = self._embedding_provider.get_vector_name()
        await self._client.upsert(
            collection_name=self._collection_name,
            points=[
                models.PointStruct(
                    id=uuid.uuid4().hex,
                    vector={vector_name: embeddings[0]},
                    payload=payload,
                )
            ],
        )

    async def find_memories(
        self, 
        query: str, 
        limit: int = 10, 
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find memories in the Qdrant collection. If there are no memories found, an empty list is returned.
        :param query: The query to use for the search.
        :param limit: Maximum number of results to return.
        :param filter_metadata: Optional metadata filter to apply to the search.
        :return: A list of dictionaries containing document content and metadata.
        """
        collection_exists = await self._client.collection_exists(self._collection_name)
        if not collection_exists:
            return []

        # Embed the query
        query_vector = await self._embedding_provider.embed_query(query)
        vector_name = self._embedding_provider.get_vector_name()

        # Prepare filter if needed
        filter_condition = None
        if filter_metadata:
            filter_condition = models.Filter(
                must=[
                    models.FieldCondition(
                        key=f"metadata.{key}",
                        match=models.MatchValue(value=value)
                    )
                    for key, value in filte`r_metadata.items()
                ]
            )

        # Search in Qdrant
        search_results = await self._client.search(
            collection_name=self._collection_name,
            query_vector=models.NamedVector(name=vector_name, vector=query_vector),
            limit=limit,
            query_filter=filter_condition,
        )

        # Return both document and metadata if available
        return [
            {
                "document": result.payload["document"],
                "metadata": result.payload.get("metadata", {}),
                "score": result.score
            } 
            for result in search_results
        ]
