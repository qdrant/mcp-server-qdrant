import logging
import uuid

# Force DEBUG logging for this module to ensure detailed logs are captured
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# If you have a global logging configuration, you might also need to ensure
# that the handlers (e.g., StreamHandler) are also set to DEBUG or a lower level.
# For simplicity here, we assume the default root logger's handlers will pick this up
# or that a handler is configured elsewhere that respects this level.
# Example to add a basic handler if none is configured:
# if not logger.handlers:
#     handler = logging.StreamHandler()
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     handler.setFormatter(formatter)
#     logger.addHandler(handler)

from typing import Any, Dict, Optional
from typing import Any, Dict, Optional

from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient, models

from mcp_server_qdrant.embeddings.base import EmbeddingProvider

# logger is already defined and configured above

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
        if not collection_name:
            logger.error("Search failed: Collection name is not specified and no default is set.")
            return []

        # Ensure the collection exists and is properly configured for the current embedding provider
        try:
            logger.debug(f"Ensuring collection '{collection_name}' exists and is configured before search.")
            await self._ensure_collection_exists(collection_name)
        except Exception as e:
            logger.error(
                f"Failed to ensure collection '{collection_name}' exists and is configured: {e}. Search cannot proceed."
            )
            return []

        # Embed the query
        # ToDo: instead of embedding text explicitly, use `models.Document`,
        # it should unlock usage of server-side inference.
        query_vector = await self._embedding_provider.embed_query(query)
        vector_name = self._embedding_provider.get_vector_name()
        logger.debug(f"Searching in collection '{collection_name}' using vector_name '{vector_name}'.")

        # Search in Qdrant
        try:
            search_results = await self._client.query_points(
                collection_name=collection_name,
                query_vector=query_vector,  # Corrected parameter name
                using=vector_name, # Specifies which named vector to use for the query
                limit=limit,
            )
        except Exception as e_search:
            logger.error(
                f"Error during Qdrant search on collection '{collection_name}' with vector_name '{vector_name}': {e_search}"
            )
            # Log the raw error if possible, as Qdrant client might wrap it
            if hasattr(e_search, 'response') and hasattr(e_search.response, 'text'):
                logger.error(f"Qdrant raw error response: {e_search.response.text}")
            elif hasattr(e_search, 'json'):
                try:
                    logger.error(f"Qdrant raw error (json): {e_search.json()}")
                except: # nosec
                    pass
            return []

        return [
            Entry(
                content=result.payload["document"],
                metadata=result.payload.get("metadata"),
            )
            for result in search_results.points
        ]

    async def _ensure_collection_exists(self, collection_name: str):
        """
        Ensure that the collection exists and is configured for the current embedding provider.
        Creates the collection if it doesn't exist.
        Updates an existing collection if it doesn't have the required named vector.
        :param collection_name: The name of the collection to ensure exists and is configured.
        """
        vector_name = self._embedding_provider.get_vector_name()
        vector_size = self._embedding_provider.get_vector_size()
        required_vector_params = models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE, # Or whatever default/configurable distance you use
        )
        logger.info(
            f"Ensuring collection '{collection_name}' is configured for NAMED vector '{vector_name}' "
            f"with size {vector_size} and distance {models.Distance.COSINE}."
        )
        
        required_params_for_our_vector = models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
        target_overall_vectors_config = {vector_name: required_params_for_our_vector}

        try:
            collection_info = await self._client.get_collection(collection_name)
            logger.info(f"Collection '{collection_name}' exists. Verifying its configuration.")
            logger.debug(f"Full current collection config from Qdrant: {collection_info.config}")
            
            current_vectors_config = collection_info.config.params.vectors
            
            is_correctly_configured = False
            if isinstance(current_vectors_config, dict) and \
               vector_name in current_vectors_config and \
               current_vectors_config[vector_name].size == vector_size and \
               current_vectors_config[vector_name].distance == required_params_for_our_vector.distance:
                is_correctly_configured = True
            
            if is_correctly_configured:
                logger.info(f"Collection '{collection_name}' is ALREADY correctly configured for vector '{vector_name}'.")
                return # Collection is fine, nothing more to do

            # If not correctly configured, log critical error and raise an exception
            # to prevent proceeding with a misconfigured collection.
            error_message = (
                f"CRITICAL: Collection '{collection_name}' exists but has an INCOMPATIBLE configuration. "
                f"Current vectors_config: {current_vectors_config}. "
                f"Expected named vector '{vector_name}' with params: {required_params_for_our_vector}. "
                f"Manual intervention or reset of collection '{collection_name}' is required. "
                f"Server will not attempt to automatically delete/recreate it with this logic."
            )
            logger.error(error_message)
            raise ValueError(error_message) # Raise an error to stop the process

        except Exception as e: # Catch Qdrant client errors (like "not found") or the ValueError from above
            if isinstance(e, ValueError) and "CRITICAL: Collection" in str(e):
                raise # Re-raise our specific critical error

            err_str = str(e).lower()
            is_not_found_error = (
                "not found" in err_str or 
                "status_code=404" in err_str or 
                (hasattr(e, 'status_code') and e.status_code == 404)
            )
            if is_not_found_error:
                logger.info(f"Collection '{collection_name}' does not exist. Proceeding to create with target config: {target_overall_vectors_config}")
                try:
                    await self._client.create_collection(
                        collection_name=collection_name,
                        vectors_config=target_overall_vectors_config,
                    )
                    logger.info(f"Successfully created collection '{collection_name}' with NAMED vector config.")
                except Exception as e_create:
                    logger.error(f"Failed to create collection '{collection_name}': {e_create}")
                    if hasattr(e_create, 'response') and hasattr(e_create.response, 'text'):
                        logger.error(f"Qdrant raw create error response: {e_create.response.text}")
                    raise
            else: # Unexpected error during get_collection
                logger.error(f"Unexpected error when checking collection '{collection_name}': {e}")
                if hasattr(e, 'response') and hasattr(e.response, 'text'):
                    logger.error(f"Raw Qdrant error response: {e.response.text}")
                raise # Re-raise other unexpected errors
