import uuid
import re
from typing import Optional, List, Dict, Any, Union, Tuple, Set

from qdrant_client import AsyncQdrantClient, models

from .embeddings.base import EmbeddingProvider


class QdrantConnector:
    """
    Encapsulates the connection to a Qdrant server and all the methods to interact with it.
    :param qdrant_url: The URL of the Qdrant server.
    :param qdrant_api_key: The API key to use for the Qdrant server.
    :param collection_name: The name of the collection to use. If multi_collection_mode is True, this is used as the default collection.
    :param embedding_provider: The embedding provider to use.
    :param qdrant_local_path: The path to the storage directory for the Qdrant client, if local mode is used.
    :param multi_collection_mode: Whether to enable multi-collection mode for AI agents.
    :param collection_prefix: Prefix for all collections in multi-collection mode.
    :param protected_collections: Set of collection names that are protected from modification or deletion.
    """

    # Regex pattern for valid collection names
    VALID_COLLECTION_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{1,64}$')
    # Separator for memory IDs (collection:id)
    MEMORY_ID_SEPARATOR = ":"

    def __init__(
        self,
        qdrant_url: Optional[str],
        qdrant_api_key: Optional[str],
        collection_name: str,
        embedding_provider: EmbeddingProvider,
        qdrant_local_path: Optional[str] = None,
        multi_collection_mode: bool = False,
        collection_prefix: Optional[str] = None,
        protected_collections: Optional[Set[str]] = None,
    ):
        self._qdrant_url = qdrant_url.rstrip("/") if qdrant_url else None
        self._qdrant_api_key = qdrant_api_key
        self._collection_name = collection_name
        self._embedding_provider = embedding_provider
        self._qdrant_local_path = qdrant_local_path
        self._multi_collection_mode = multi_collection_mode
        self._collection_prefix = collection_prefix or ""
        self._protected_collections = protected_collections or set()
        
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
        # If not in multi-collection mode or collection_name is None, use the default collection name
        if not self._multi_collection_mode:
            return self._collection_name
            
        # If collection_name is None or empty, use the default collection name
        if not collection_name:
            return self._collection_name
        
        # If collection_name is the default collection name, return it as is
        if collection_name == self._collection_name:
            return self._collection_name
        
        # Validate the collection name
        if not self.VALID_COLLECTION_NAME_PATTERN.match(collection_name):
            raise ValueError(
                f"Invalid collection name: {collection_name}. "
                "Collection names must contain only alphanumeric characters, underscores, and hyphens, "
                "and be between 1 and 64 characters long."
            )
        
        return f"{self._collection_prefix}{collection_name}"

    def _get_collection_from_memory_id(self, memory_id: str) -> Tuple[str, str]:
        """
        Extract the collection name and point ID from a memory ID.
        :param memory_id: The memory ID in the format "collection:id"
        :return: A tuple of (collection_name, point_id)
        """
        if self.MEMORY_ID_SEPARATOR not in memory_id:
            # If no separator, assume it's in the default collection
            return self._collection_name, memory_id
            
        collection, point_id = memory_id.split(self.MEMORY_ID_SEPARATOR, 1)
        return collection, point_id

    def _create_memory_id(self, collection_name: str, point_id: str) -> str:
        """
        Create a memory ID from a collection name and point ID.
        :param collection_name: The collection name
        :param point_id: The point ID
        :return: A memory ID in the format "collection:id"
        """
        # For the default collection in single-collection mode, we can just return the point ID
        if collection_name == self._collection_name and not self._multi_collection_mode:
            return point_id
        return f"{collection_name}{self.MEMORY_ID_SEPARATOR}{point_id}"

    def _is_collection_protected(self, collection_name: str) -> bool:
        """
        Check if a collection is protected from modification or deletion.
        :param collection_name: The collection name to check
        :return: True if the collection is protected, False otherwise
        """
        # If "*" is in protected_collections, all collections are protected
        if "*" in self._protected_collections:
            return True
            
        return collection_name in self._protected_collections

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
        # if for some reason we're not in multi-collection mode, return the default collection
        if not self._multi_collection_mode:
            return [self._collection_name]

        collections = await self._client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        # Ensure default collection is first in the list if it exists
        if self._collection_name in collection_names:
            collection_names.remove(self._collection_name)

        # always make sure the default collection is first in the list
        collection_names.insert(0, self._collection_name)
        
        # If in multi-collection mode, filter to only show collections with the prefix
        if self._multi_collection_mode and self._collection_prefix:
            result = [self._collection_name] # always include the default collection

            prefix_len = len(self._collection_prefix)
            result += [
                name[prefix_len:] 
                for name in collection_names 
                if name.startswith(self._collection_prefix) and name != self._collection_name
            ]
                
            return result

        # If in multi-collection mode, and no prefix is set, return all collections
        return collection_names

    async def store_memory(
        self, 
        information: str, 
        collection_name: Optional[str] = None,
        replace_memory_ids: Optional[List[str]] = None
    ) -> str:
        """
        Store a memory in the Qdrant collection.
        :param information: The information to store.
        :param collection_name: The name of the collection to store the memory in. 
                               If None, use the default collection.
        :param replace_memory_ids: Optional list of memory IDs to delete before storing the new memory.
        :return: The ID of the stored memory.
        :raises ValueError: If the collection is protected from modification.
        """
        collection_for_id = collection_name if collection_name else self._collection_name
        
        # Check if the collection is protected
        if self._is_collection_protected(collection_for_id):
            raise ValueError(f"Collection '{collection_for_id}' is protected and cannot be modified.")
            
        # Delete memories to be replaced if specified
        if replace_memory_ids:
            await self.delete_memories(replace_memory_ids)
            
        await self._ensure_collection_exists(collection_name)
        prefixed_name = self._get_prefixed_collection_name(collection_name)

        # Embed the document
        embeddings = await self._embedding_provider.embed_documents([information])

        # Generate a unique ID for the point
        point_id = uuid.uuid4().hex

        # Add to Qdrant
        vector_name = self._embedding_provider.get_vector_name()
        await self._client.upsert(
            collection_name=prefixed_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector={vector_name: embeddings[0]},
                    payload={"document": information},
                )
            ],
        )
        
        # Return the memory ID
        return self._create_memory_id(collection_for_id, point_id)

    async def find_memories(
        self, 
        query: str, 
        collection_name: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Find memories in the Qdrant collection. If there are no memories found, an empty list is returned.
        :param query: The query to use for the search.
        :param collection_name: The name of the collection to search in. 
                               If None, use the default collection.
        :return: A list of dictionaries containing the memory content and ID.
        """
        prefixed_name = self._get_prefixed_collection_name(collection_name)
        collection_exists = await self._client.collection_exists(prefixed_name)
        collection_for_id = collection_name if collection_name else self._collection_name
        
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

        # Return memories with their IDs
        return [
            {
                "content": result.payload["document"],
                "id": self._create_memory_id(collection_for_id, result.id),
                "readonly": self._is_collection_protected(collection_for_id)
            } 
            for result in search_results
        ]

    async def find_memories_across_collections(self, query: str) -> Dict[str, List[Dict[str, str]]]:
        """
        Find memories across all collections available to the AI agent.
        :param query: The query to use for the search.
        :return: A dictionary mapping collection names to lists of memories found.
        """
        collections = await self.list_collections()
        results = {}
        
        # Always include the default collection
        default_memories = await self.find_memories(query)
        if default_memories:
            results[self._collection_name] = default_memories
        
        # Search in each collection
        for collection in collections:
            if collection == self._collection_name:
                continue  # Skip the default collection as we've already searched it
                
            memories = await self.find_memories(query, collection)
            if memories:
                results[collection] = memories
                
        return results
        
    async def delete_memories(self, memory_ids: List[str]) -> int:
        """
        Delete memories by their IDs.
        :param memory_ids: List of memory IDs to delete.
        :return: Number of memories successfully deleted.
        :raises ValueError: If any of the collections are protected from deletion.
        """
        # Group memory IDs by collection for batch deletion
        collection_points: Dict[str, List[str]] = {}
        
        for memory_id in memory_ids:
            collection, point_id = self._get_collection_from_memory_id(memory_id)
            
            # Check if the collection is protected
            if self._is_collection_protected(collection):
                raise ValueError(f"Collection '{collection}' is protected and cannot be modified.")
                
            # Get the prefixed collection name - this handles the default collection correctly
            prefixed_name = self._get_prefixed_collection_name(collection)
            
            if prefixed_name not in collection_points:
                collection_points[prefixed_name] = []
                
            collection_points[prefixed_name].append(point_id)
        
        # Delete points from each collection
        deleted_count = 0
        for collection_name, point_ids in collection_points.items():
            # Check if collection exists
            collection_exists = await self._client.collection_exists(collection_name)
            if not collection_exists:
                continue
                
            # Delete points
            result = await self._client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(
                    points=point_ids
                )
            )
            
            # The UpdateStatus object doesn't have a deleted_count attribute
            # Instead, we'll assume all points were deleted if the operation was successful
            # In a production environment, you might want to verify this by checking if the points still exist
            deleted_count += len(point_ids)
            
        return deleted_count
