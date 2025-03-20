import json
import uuid
import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, TypeAlias, Union

from qdrant_client import AsyncQdrantClient, models

from .embeddings.base import EmbeddingProvider

# Type alias for metadata
Metadata: TypeAlias = Dict[str, Any]

logger = logging.getLogger(__name__)


@dataclass
class Entry:
    """
    Represents an entry in the Qdrant collection.
    :param content: The content of the entry.
    :param metadata: Optional metadata associated with the entry.
    :param id: Optional ID for the entry. If not provided, a random UUID will be generated.
    """
    content: str
    metadata: Optional[Metadata] = None
    id: Optional[str] = None


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
            logger.info(f"Created collection {self._collection_name} with vector size {vector_size}")

    async def store(self, entry: Entry):
        """
        Store an entry in the Qdrant collection.
        :param entry: The entry to store.
        """
        await self._ensure_collection_exists()

        # Generate ID if not provided
        entry_id = entry.id or uuid.uuid4().hex

        # Embed the document
        embeddings = await self._embedding_provider.embed_documents([entry.content])

        # Add to Qdrant
        vector_name = self._embedding_provider.get_vector_name()
        payload = {"document": entry.content}
        
        # Add metadata to payload if provided
        if entry.metadata:
            payload["metadata"] = entry.metadata

        await self._client.upsert(
            collection_name=self._collection_name,
            points=[
                models.PointStruct(
                    id=entry_id,
                    vector={vector_name: embeddings[0]},
                    payload=payload,
                )
            ],
        )
        logger.debug(f"Stored entry with ID {entry_id}")

    async def store_memory(self, information: str):
        """
        Legacy method to store a memory in the Qdrant collection.
        :param information: The information to store.
        """
        entry = Entry(content=information)
        await self.store(entry)

    async def search(self, query: str, limit: int = 10) -> List[Entry]:
        """
        Search for entries in the Qdrant collection.
        :param query: The query to use for the search.
        :param limit: The maximum number of results to return.
        :return: A list of entries found.
        """
        collection_exists = await self._client.collection_exists(self._collection_name)
        if not collection_exists:
            return []

        # Embed the query
        query_vector = await self._embedding_provider.embed_query(query)
        vector_name = self._embedding_provider.get_vector_name()

        # Search in Qdrant
        search_results = await self._client.search(
            collection_name=self._collection_name,
            query_vector=models.NamedVector(name=vector_name, vector=query_vector),
            limit=limit,
        )

        # Convert results to Entry objects
        entries = []
        for result in search_results:
            content = result.payload.get("document", "")
            metadata = result.payload.get("metadata")
            entry_id = str(result.id)
            entries.append(Entry(content=content, metadata=metadata, id=entry_id))

        return entries

    async def find_memories(self, query: str) -> list[str]:
        """
        Legacy method to find memories in the Qdrant collection.
        :param query: The query to use for the search.
        :return: A list of memory contents found.
        """
        entries = await self.search(query)
        return [entry.content for entry in entries]
