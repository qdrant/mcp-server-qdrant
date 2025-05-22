import logging
import uuid
from typing import Any, Dict, Optional

from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient, models

from mcp_server_qdrant.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)

Metadata = Dict[str, Any]


class Entry(BaseModel):
    """
    A single entry in the Qdrant collection.
    """

    content: str
    metadata: Optional[Metadata] = None


class QdrantConnector:
    """
    Encapsulates the connection to a Qdrant server and all the methods to interact with it.
    :param qdrant_url: The URL of the Qdrant server.
    :param qdrant_api_key: The API key to use for the Qdrant server.
    :param collection_name: The name of the default collection to use. If not provided, each tool will require
                            the collection name to be provided.
    :param embedding_provider: The embedding provider to use.
    :param qdrant_local_path: The path to the storage directory for the Qdrant client, if local mode is used.
    """

    def __init__(
        self,
        qdrant_url: Optional[str],
        qdrant_api_key: Optional[str],
        collection_name: Optional[str],
        embedding_provider: EmbeddingProvider,
        qdrant_local_path: Optional[str] = None,
    ):
        self._qdrant_url = qdrant_url.rstrip("/") if qdrant_url else None
        self._qdrant_api_key = qdrant_api_key
        self._default_collection_name = collection_name
        self._embedding_provider = embedding_provider
        self._client = AsyncQdrantClient(
            location=qdrant_url, api_key=qdrant_api_key, path=qdrant_local_path
        )

    async def get_collection_names(self) -> list[str]:
        """
        Get the names of all collections in the Qdrant server.
        :return: A list of collection names.
        """
        response = await self._client.get_collections()
        return [collection.name for collection in response.collections]

    async def store(self, entry: Entry, *, collection_name: Optional[str] = None):
        """
        Store some information in the Qdrant collection, along with the specified metadata.
        :param entry: The entry to store in the Qdrant collection.
        :param collection_name: The name of the collection to store the information in, optional. If not provided,
                                the default collection is used.
        """
        collection_name = collection_name or self._default_collection_name
        assert collection_name is not None
        await self._ensure_collection_exists(collection_name)

        # Embed the document
        # ToDo: instead of embedding text explicitly, use `models.Document`,
        # it should unlock usage of server-side inference.
        embeddings = await self._embedding_provider.embed_documents([entry.content])

        # Add to Qdrant
        vector_name = self._embedding_provider.get_vector_name()
        payload = {"document": entry.content, "metadata": entry.metadata}
        await self._client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=uuid.uuid4().hex,
                    vector={vector_name: embeddings[0]},
                    payload=payload,
                )
            ],
        )

    async def search(
        self, query: str, *, collection_name: Optional[str] = None, limit: int = 10
    ) -> list[Entry]:
        """
        Find points in the Qdrant collection. If there are no entries found, an empty list is returned.
        :param query: The query to use for the search.
        :param collection_name: The name of the collection to search in, optional. If not provided,
                                the default collection is used.
        :param limit: The maximum number of entries to return.
        :return: A list of entries found.
        """
        collection_name = collection_name or self._default_collection_name
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            return []

        # Embed the query
        # ToDo: instead of embedding text explicitly, use `models.Document`,
        # it should unlock usage of server-side inference.

        query_vector = await self._embedding_provider.embed_query(query)
        
        # Check if this is a legacy collection (single vector) or named vectors
        is_legacy_collection = await self._is_legacy_collection(collection_name)
        
        # Search in Qdrant
        if is_legacy_collection:
            # Legacy collections don't use named vectors
            search_results = await self._client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=limit,
            )
        else:
            # Modern collections use named vectors
            vector_name = self._embedding_provider.get_vector_name()
            search_results = await self._client.query_points(
                collection_name=collection_name,
                query=query_vector,
                using=vector_name,
                limit=limit,
            )

        entries = []
        for result in search_results.points:
            if result.payload:
                # Try to get content from "document" key, then "content" key
                doc_content = result.payload.get("document")
                if doc_content is None:
                    doc_content = result.payload.get("content", "Error: Neither 'document' nor 'content' key found in payload")
                
                entry_metadata = result.payload.get("metadata")
                entries.append(Entry(content=doc_content, metadata=entry_metadata))
            else:
                # Optionally log points with no payload or handle as an error
                entries.append(Entry(content="Error: Point has no payload", metadata=None))
        return entries

    async def _is_legacy_collection(self, collection_name: str) -> bool:
        """
        Check if a collection uses the legacy single vector format (pre-named vectors).
        :param collection_name: The name of the collection to check.
        :return: True if it's a legacy collection, False if it uses named vectors.
        """
        try:
            collection_info = await self._client.get_collection(collection_name)
            vectors_config = collection_info.config.params.vectors
            
            # Legacy collections have vectors config with 'size' and 'distance' directly
            # Named vector collections have a dict of vector names
            if hasattr(vectors_config, 'size') and hasattr(vectors_config, 'distance'):
                return True
            elif isinstance(vectors_config, dict):
                # If it's a dict, check if any values have 'size' directly (legacy)
                # or if all values are VectorParams objects (modern named vectors)
                for key, value in vectors_config.items():
                    if hasattr(value, 'size') and hasattr(value, 'distance'):
                        # This is a VectorParams object, so modern named vectors
                        return False
                    elif isinstance(value, dict) and 'size' in value and 'distance' in value:
                        # This looks like legacy format embedded in a dict
                        return True
                # If we have a dict but couldn't determine format, assume modern
                return False
            else:
                # Unknown format, assume modern
                return False
        except Exception as e:
            logger.warning(f"Could not determine collection format for {collection_name}: {e}")
            # Default to legacy format for safety
            return True

    async def _ensure_collection_exists(self, collection_name: str):
        """
        Ensure that the collection exists, creating it if necessary.
        :param collection_name: The name of the collection to ensure exists.
        """
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            # Create the collection with the appropriate vector size
            vector_size = self._embedding_provider.get_vector_size()

            # Use the vector name as defined in the embedding provider
            vector_name = self._embedding_provider.get_vector_name()
            await self._client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    vector_name: models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                    )
                },
            )
