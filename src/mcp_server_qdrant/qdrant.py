import logging
import uuid
from typing import Any

from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient, models

from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.settings import METADATA_PATH

logger = logging.getLogger(__name__)

Metadata = dict[str, Any]
ArbitraryFilter = dict[str, Any]


class Entry(BaseModel):
    """
    A single entry in the Qdrant collection.
    """

    content: str
    metadata: Metadata | None = None


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
        qdrant_url: str | None,
        qdrant_api_key: str | None,
        collection_name: str | None,
        embedding_provider: EmbeddingProvider,
        qdrant_local_path: str | None = None,
        field_indexes: dict[str, models.PayloadSchemaType] | None = None,
    ):
        self._qdrant_url = qdrant_url.rstrip("/") if qdrant_url else None
        self._qdrant_api_key = qdrant_api_key
        self._default_collection_name = collection_name
        self._embedding_provider = embedding_provider
        self._client = AsyncQdrantClient(
            location=qdrant_url, api_key=qdrant_api_key, path=qdrant_local_path
        )
        self._field_indexes = field_indexes

    async def get_collection_names(self) -> list[str]:
        """
        Get the names of all collections in the Qdrant server.
        :return: A list of collection names.
        """
        response = await self._client.get_collections()
        return [collection.name for collection in response.collections]

    async def get_collection_info(self, collection_name: str) -> dict[str, Any]:
        """
        Get detailed information about a specific collection.
        :param collection_name: The name of the collection.
        :return: Collection information as a dictionary.
        """
        info = await self._client.get_collection(collection_name)
        return {
            "status": info.status.name if hasattr(info.status, "name") else str(info.status),
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "config": {
                "params": str(info.config.params) if info.config.params else None,
            },
        }

    async def get_point(
        self, point_id: str, collection_name: str | None = None
    ) -> Entry | None:
        """
        Get a specific point by ID from a collection.
        :param point_id: The ID of the point to retrieve.
        :param collection_name: The name of the collection, optional.
        :return: The entry if found, None otherwise.
        """
        collection_name = collection_name or self._default_collection_name
        assert collection_name is not None

        try:
            result = await self._client.retrieve(
                collection_name=collection_name,
                ids=[point_id],
                with_payload=True,
                with_vectors=False,
            )
            if result:
                point = result[0]
                return Entry(
                    content=point.payload.get("document") or point.payload.get("text"),
                    metadata=point.payload.get("metadata"),
                )
            return None
        except Exception as e:
            logger.error(f"Error retrieving point {point_id}: {e}")
            return None

    async def delete_point(
        self, point_id: str, collection_name: str | None = None
    ) -> bool:
        """
        Delete a specific point by ID from a collection.
        :param point_id: The ID of the point to delete.
        :param collection_name: The name of the collection, optional.
        :return: True if deleted successfully, False otherwise.
        """
        collection_name = collection_name or self._default_collection_name
        assert collection_name is not None

        try:
            await self._client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(points=[point_id]),
            )
            return True
        except Exception as e:
            logger.error(f"Error deleting point {point_id}: {e}")
            return False

    async def update_point_payload(
        self,
        point_id: str,
        metadata: Metadata,
        collection_name: str | None = None,
    ) -> bool:
        """
        Update the metadata/payload for a specific point.
        :param point_id: The ID of the point to update.
        :param metadata: The new metadata to set.
        :param collection_name: The name of the collection, optional.
        :return: True if updated successfully, False otherwise.
        """
        collection_name = collection_name or self._default_collection_name
        assert collection_name is not None

        try:
            await self._client.set_payload(
                collection_name=collection_name,
                payload={METADATA_PATH: metadata},
                points=[point_id],
            )
            return True
        except Exception as e:
            logger.error(f"Error updating payload for point {point_id}: {e}")
            return False

    async def store(self, entry: Entry, *, collection_name: str | None = None):
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
        # Include both "text" and "document" for compatibility
        payload = {
            "text": entry.content,
            "document": entry.content,
            METADATA_PATH: entry.metadata,
        }

        # Handle unnamed vectors differently
        if vector_name == "unnamed_vector":
            await self._client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=uuid.uuid4().hex,
                        vector=embeddings[0],  # Direct vector for unnamed
                        payload=payload,
                    )
                ],
            )
        else:
            await self._client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=uuid.uuid4().hex,
                        vector={vector_name: embeddings[0]},  # Named vector dict
                        payload=payload,
                    )
                ],
            )

    async def search(
        self,
        query: str,
        *,
        collection_name: str | None = None,
        limit: int = 10,
        query_filter: models.Filter | None = None,
    ) -> list[Entry]:
        """
        Find points in the Qdrant collection. If there are no entries found, an empty list is returned.
        :param query: The query to use for the search.
        :param collection_name: The name of the collection to search in, optional. If not provided,
                                the default collection is used.
        :param limit: The maximum number of entries to return.
        :param query_filter: The filter to apply to the query, if any.

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
        vector_name = self._embedding_provider.get_vector_name()

        # Search in Qdrant with conditional logic for unnamed vectors
        if vector_name == "unnamed_vector":
            search_results = await self._client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=limit,
                query_filter=query_filter,
            )
        else:
            search_results = await self._client.query_points(
                collection_name=collection_name,
                query=query_vector,
                using=vector_name,
                limit=limit,
                query_filter=query_filter,
            )

        return [
            Entry(
                content=result.payload.get("document") or result.payload.get("text"),
                metadata=result.payload.get("metadata"),
            )
            for result in search_results.points
        ]

    async def find_hybrid(
        self,
        query: str,
        *,
        collection_name: str | None = None,
        fusion_method: str = "rrf",
        dense_limit: int = 10,
        sparse_limit: int = 10,
        final_limit: int = 10,
        query_filter: models.Filter | None = None,
    ) -> list[Entry]:
        """
        Find points in the Qdrant collection using hybrid search (dense + sparse vectors).
        :param query: The query to use for the search.
        :param collection_name: The name of the collection to search in.
        :param fusion_method: Fusion method to use ('rrf' or 'dbsf').
        :param dense_limit: Limit for dense vector prefetch.
        :param sparse_limit: Limit for sparse vector prefetch.
        :param final_limit: Final limit after fusion.
        :param query_filter: The filter to apply to the query, if any.
        :return: A list of entries found.
        """
        collection_name = collection_name or self._default_collection_name
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            return []

        # Embed the query for dense vector search
        query_vector = await self._embedding_provider.embed_query(query)
        vector_name = self._embedding_provider.get_vector_name()

        # Prepare dense prefetch query
        dense_prefetch = models.Prefetch(
            query=query_vector,
            using=vector_name if vector_name != "unnamed_vector" else None,
            limit=dense_limit,
        )

        # Try to use sparse vectors if available
        try:
            # Prepare sparse prefetch query
            # Note: Sparse vector name is typically "sparse-text" or similar
            sparse_prefetch = models.Prefetch(
                query=models.SparseVector(
                    indices=[],  # Will be populated by Qdrant's sparse encoder
                    values=[],
                ),
                using="sparse-text",  # Default sparse vector name
                limit=sparse_limit,
            )

            # Set up fusion method
            if fusion_method.lower() == "rrf":
                fusion = models.Fusion.RRF
            elif fusion_method.lower() == "dbsf":
                fusion = models.Fusion.DBSF
            else:
                logger.warning(
                    f"Unknown fusion method: {fusion_method}, defaulting to RRF"
                )
                fusion = models.Fusion.RRF

            # Execute hybrid search with both prefetches
            search_results = await self._client.query_points(
                collection_name=collection_name,
                prefetch=[dense_prefetch, sparse_prefetch],
                query=fusion,
                limit=final_limit,
                query_filter=query_filter,
            )
        except Exception as e:
            # Fallback to dense-only search if sparse vectors are not available
            logger.debug(
                f"Sparse vectors not available, falling back to dense-only search: {e}"
            )
            if vector_name == "unnamed_vector":
                search_results = await self._client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    limit=final_limit,
                    query_filter=query_filter,
                )
            else:
                search_results = await self._client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    using=vector_name,
                    limit=final_limit,
                    query_filter=query_filter,
                )

        return [
            Entry(
                content=result.payload.get("document") or result.payload.get("text"),
                metadata=result.payload.get("metadata"),
            )
            for result in search_results.points
        ]

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

            # Handle unnamed vectors differently
            if vector_name == "unnamed_vector":
                await self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                    ),
                )
            else:
                await self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        vector_name: models.VectorParams(
                            size=vector_size,
                            distance=models.Distance.COSINE,
                        )
                    },
                )

            # Create payload indexes if configured

            if self._field_indexes:
                for field_name, field_type in self._field_indexes.items():
                    await self._client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field_name,
                        field_schema=field_type,
                    )
