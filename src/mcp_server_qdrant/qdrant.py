import uuid
import re
from typing import Optional, List, Dict, Any

from qdrant_client import AsyncQdrantClient, models

from .embeddings.base import EmbeddingProvider


class QdrantConnector:
    """
    Encapsulates the connection to a Qdrant server and all the methods to interact with it.
    :param qdrant_url: The URL of the Qdrant server.
    :param qdrant_api_key: The API key to use for the Qdrant server.
    :param collection_name: The name of the collection to use. If multi_collection_mode is True, this is used as the global collection.
    :param embedding_provider: The embedding provider to use.
    :param qdrant_local_path: The path to the storage directory for the Qdrant client, if local mode is used.
    :param multi_collection_mode: Whether to enable multi-collection mode for AI agents.
    :param collection_prefix: Prefix for all collections in multi-collection mode.
    """

    # Regex pattern for valid collection names
    VALID_COLLECTION_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{1,64}$')

    def __init__(
        self,
        qdrant_url: Optional[str],
        qdrant_api_key: Optional[str],
        collection_name: str,
        embedding_provider: EmbeddingProvider,
        qdrant_local_path: Optional[str] = None,
        multi_collection_mode: bool = False,
        collection_prefix: Optional[str] = None,
    ):
        self._qdrant_url = qdrant_url.rstrip("/") if qdrant_url else None
        self._qdrant_api_key = qdrant_api_key
        self._collection_name = collection_name
        self._embedding_provider = embedding_provider
        self._qdrant_local_path = qdrant_local_path
        self._multi_collection_mode = multi_collection_mode
        self._collection_prefix = collection_prefix or ""
        
        # Initialize the Qdrant client
        if qdrant_url:
            self._client = AsyncQdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key,
            )
        else:
            self._client = AsyncQdrantClient(
                path=qdrant_local_path or ":memory:",
            )

    def _get_prefixed_collection_name(self, collection_name: Optional[str] = None) -> str:
        """
        Get the prefixed collection name.
        :param collection_name: The collection name to prefix. If None, use the default collection name.
        :return: The prefixed collection name.
        """
        if not self._multi_collection_mode or not collection_name:
            return self._collection_name
        
        # Validate the collection name
        if not self.VALID_COLLECTION_NAME_PATTERN.match(collection_name):
            raise ValueError(
                f"Invalid collection name: {collection_name}. "
                "Collection names must contain only alphanumeric characters, underscores, and hyphens, "
                "and be between 1 and 64 characters long."
            )
        
        return f"{self._collection_prefix}{collection_name}"

    async def _ensure_collection_exists(self, collection_name: Optional[str] = None):
        """
        Ensure that the collection exists, creating it if necessary.
        :param collection_name: The name of the collection to ensure exists. If None, use the default collection name.
        """
        prefixed_name = self._get_prefixed_collection_name(collection_name)
        collection_exists = await self._client.collection_exists(prefixed_name)
        
        if not collection_exists:
            # Create the collection with the appropriate vector size
            # We'll get the vector size by embedding a sample text
            sample_vector = await self._embedding_provider.embed_query("sample text")
            vector_size = len(sample_vector)

            # Use the vector name as defined in the embedding provider
            vector_name = self._embedding_provider.get_vector_name()
            
            # Create default vector parameters
            vector_params = models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            )
            
            # Create the collection with the default parameters
            await self._client.create_collection(
                collection_name=prefixed_name,
                vectors_config={
                    vector_name: vector_params
                }
            )

    async def list_collections(self) -> List[str]:
        """
        List all collections available to the AI agent.
        :return: A list of collection names (without the prefix).
        """
        collections = await self._client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        # If in multi-collection mode, filter to only show collections with the prefix
        if self._multi_collection_mode and self._collection_prefix:
            prefix_len = len(self._collection_prefix)
            return [
                name[prefix_len:] 
                for name in collection_names 
                if name.startswith(self._collection_prefix)
            ]
        
        return collection_names

    async def store_memory(self, information: str, collection_name: Optional[str] = None):
        """
        Store a memory in the Qdrant collection.
        :param information: The information to store.
        :param collection_name: The name of the collection to store the memory in. 
                               If None, use the default collection.
        """
        await self._ensure_collection_exists(collection_name)
        prefixed_name = self._get_prefixed_collection_name(collection_name)

        # Embed the document
        embeddings = await self._embedding_provider.embed_documents([information])

        # Add to Qdrant
        vector_name = self._embedding_provider.get_vector_name()
        await self._client.upsert(
            collection_name=prefixed_name,
            points=[
                models.PointStruct(
                    id=uuid.uuid4().hex,
                    vector={vector_name: embeddings[0]},
                    payload={"document": information},
                )
            ],
        )

    async def find_memories(self, query: str, collection_name: Optional[str] = None) -> list[str]:
        """
        Find memories in the Qdrant collection. If there are no memories found, an empty list is returned.
        :param query: The query to use for the search.
        :param collection_name: The name of the collection to search in. 
                               If None, use the default collection.
        :return: A list of memories found.
        """
        prefixed_name = self._get_prefixed_collection_name(collection_name)
        collection_exists = await self._client.collection_exists(prefixed_name)
        
        if not collection_exists:
            return []

        # Embed the query
        query_vector = await self._embedding_provider.embed_query(query)
        vector_name = self._embedding_provider.get_vector_name()

        # Search in Qdrant
        search_results = await self._client.search(
            collection_name=prefixed_name,
            query_vector=models.NamedVector(name=vector_name, vector=query_vector),
            limit=10,
        )

        return [result.payload["document"] for result in search_results]

    async def find_memories_across_collections(self, query: str) -> Dict[str, List[str]]:
        """
        Find memories across all collections available to the AI agent.
        :param query: The query to use for the search.
        :return: A dictionary mapping collection names to lists of memories found.
        """
        if not self._multi_collection_mode:
            memories = await self.find_memories(query)
            return {self._collection_name: memories}
        
        collections = await self.list_collections()
        results = {}
        
        # Always include the global collection
        global_memories = await self.find_memories(query)
        if global_memories:
            results["global"] = global_memories
        
        # Search in each collection
        for collection in collections:
            if collection == self._collection_name:
                continue  # Skip the global collection as we've already searched it
                
            memories = await self.find_memories(query, collection)
            if memories:
                results[collection] = memories
                
        return results
