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
            f"Ensuring collection '{collection_name}' is configured for vector_name '{vector_name}' "
            f"with size {vector_size}."
        )

        try:
            collection_info = await self._client.get_collection(collection_name)
            logger.info(f"Collection '{collection_name}' found. Current config: {collection_info.config}")
            current_vectors_config = collection_info.config.params.vectors
            logger.debug(f"Current vectors_config for '{collection_name}': {current_vectors_config}")

            needs_update = False
            if isinstance(current_vectors_config, dict):  # Named vectors config
                if vector_name not in current_vectors_config:
                    logger.info(
                        f"Named vector '{vector_name}' not found in config of collection '{collection_name}'. "
                        f"Existing named vectors: {list(current_vectors_config.keys())}."
                    )
                    needs_update = True
                else:
                    existing_params = current_vectors_config[vector_name]
                    logger.info(f"Named vector '{vector_name}' found in '{collection_name}'. Params: {existing_params}")
                    if existing_params.size != vector_size:
                        logger.warning(
                            f"Vector '{vector_name}' in collection '{collection_name}' has mismatched size. "
                            f"Expected {vector_size}, got {existing_params.size}. This may cause issues. "
                            f"Attempting to update."
                        )
                        needs_update = True # Qdrant might not allow direct size update, this might require recreation or careful handling
                    # Add more checks if needed (e.g., distance function)
            elif isinstance(current_vectors_config, models.VectorParams):  # Legacy single unnamed vector
                logger.warning(
                    f"Collection '{collection_name}' uses a legacy unnamed vector configuration. "
                    f"It needs to be updated for named vector '{vector_name}'."
                )
                needs_update = True
            else:
                logger.error(
                    f"Unrecognized vectors configuration type for collection '{collection_name}': {type(current_vectors_config)}. "
                    f"Cannot determine if update is needed."
                )
                # Depending on strictness, you might want to raise an error here or simply return
                return

            if needs_update:
                logger.info(
                    f"Attempting to update collection '{collection_name}' to include/update vector '{vector_name}' "
                    f"with params: {required_vector_params}."
                )
                try:
                    update_vectors_config = {vector_name: required_vector_params}
                    logger.info(f"Calling update_collection for '{collection_name}' with vectors_config: {update_vectors_config}")
                    # For named vectors, update_collection expects a dict of vector names to params
                    await self._client.update_collection(
                        collection_name=collection_name,
                        vectors_config=update_vectors_config
                    )
                    logger.info(f"Successfully updated collection '{collection_name}' for vector '{vector_name}'.")
                except Exception as e_update:
                    logger.error(f"Failed to update collection '{collection_name}' for vector '{vector_name}': {e_update}")
                    # Log raw error if available
                    if hasattr(e_update, 'response') and hasattr(e_update.response, 'text'):
                        logger.error(f"Qdrant raw update error response: {e_update.response.text}")
                    elif hasattr(e_update, 'json'):
                        try:
                            logger.error(f"Qdrant raw update error (json): {e_update.json()}")
                        except: # nosec
                            pass
                    raise  # Re-raise to signal that configuration might still be an issue.

        except Exception as e:
            err_str = str(e).lower()
            is_not_found_error = (
                "not found" in err_str or
                "status_code=404" in err_str or
                (hasattr(e, 'status_code') and e.status_code == 404) or
                (isinstance(e, ValueError) and "not found" in err_str) # For some local client errors
            )

            if is_not_found_error:
                logger.info(
                    f"Collection '{collection_name}' does not exist. Creating it with vector '{vector_name}' "
                    f"and params: {required_vector_params}."
                )
                try:
                    create_vectors_config = {vector_name: required_vector_params}
                    logger.info(f"Calling create_collection for '{collection_name}' with vectors_config: {create_vectors_config}")
                    await self._client.create_collection(
                        collection_name=collection_name,
                        vectors_config=create_vectors_config,
                    )
                    logger.info(f"Successfully created collection '{collection_name}' with vector '{vector_name}'.")
                except Exception as e_create:
                    logger.error(f"Failed to create collection '{collection_name}': {e_create}")
                    if hasattr(e_create, 'response') and hasattr(e_create.response, 'text'):
                        logger.error(f"Qdrant raw create error response: {e_create.response.text}")
                    elif hasattr(e_create, 'json'):
                        try:
                            logger.error(f"Qdrant raw create error (json): {e_create.json()}")
                        except: # nosec
                            pass
                    raise
            else:
                logger.error(f"Failed to get or process collection info for '{collection_name}' (not a 'not found' error): {e}")
                if hasattr(e, 'response') and hasattr(e.response, 'text'):
                    logger.error(f"Qdrant raw get_collection error response: {e.response.text}")
                elif hasattr(e, 'json'):
                    try:
                        logger.error(f"Qdrant raw get_collection error (json): {e.json()}")
                    except: # nosec
                        pass
                raise
