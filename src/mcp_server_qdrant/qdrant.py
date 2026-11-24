import logging
import uuid
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

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


class SearchResults(list[Entry]):
    """
    The entries found by a search. It is a plain list of entries, plus the
    number of matching points that were skipped because their payload has no
    "document" field.
    """

    def __init__(self, entries: Iterable[Entry] = (), skipped: int = 0):
        super().__init__(entries)
        self.skipped = skipped


class IncompatibleCollectionError(ValueError):
    """
    The collection's vectors do not match the configured embedding model.
    """


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
        payload = {"document": entry.content, METADATA_PATH: entry.metadata}
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
        self,
        query: str,
        *,
        collection_name: str | None = None,
        limit: int = 10,
        query_filter: models.Filter | None = None,
    ) -> SearchResults:
        """
        Find points in the Qdrant collection. If there are no entries found, an empty list is returned.
        Points without a "document" field in their payload are skipped and counted in
        ``SearchResults.skipped``.
        :param query: The query to use for the search.
        :param collection_name: The name of the collection to search in, optional. If not provided,
                                the default collection is used.
        :param limit: The maximum number of entries to return.
        :param query_filter: The filter to apply to the query, if any.

        :return: The entries found.
        """
        collection_name = collection_name or self._default_collection_name
        assert collection_name is not None
        collection_info = await self._get_collection(collection_name)
        if collection_info is None:
            return SearchResults()
        self._check_vectors(collection_name, collection_info)

        # Embed the query
        # ToDo: instead of embedding text explicitly, use `models.Document`,
        # it should unlock usage of server-side inference.

        query_vector = await self._embedding_provider.embed_query(query)
        vector_name = self._embedding_provider.get_vector_name()

        # Search in Qdrant
        search_results = await self._client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using=vector_name,
            limit=limit,
            query_filter=query_filter,
        )

        results = SearchResults()
        for point in search_results.points:
            payload = point.payload or {}
            if "document" not in payload:
                results.skipped += 1
                continue
            results.append(
                Entry(content=payload["document"], metadata=payload.get("metadata"))
            )
        if results.skipped:
            logger.warning(
                "Skipped %d point(s) without a 'document' payload field in collection %r",
                results.skipped,
                collection_name,
            )
        return results

    async def _get_collection(
        self, collection_name: str
    ) -> models.CollectionInfo | None:
        """
        Get the collection's info, or None if it does not exist. This replaces a
        separate existence check, so validating the collection costs no extra request.
        """
        try:
            return await self._client.get_collection(collection_name)
        except UnexpectedResponse as e:
            # Remote mode: the server returns 404 for a missing collection.
            if e.status_code == 404:
                return None
            raise
        except ValueError as e:
            # Local mode raises ValueError("Collection ... not found").
            if "not found" in str(e).lower():
                return None
            raise

    def _check_vectors(
        self, collection_name: str, collection_info: models.CollectionInfo
    ) -> None:
        """
        Raise a clear error if the collection has no vector with the name and size
        that the embedding model produces. Without this check, Qdrant rejects the
        request with a cryptic "Not existing vector name" or dimension error.
        """
        expected_name = self._embedding_provider.get_vector_name()
        expected_size = self._embedding_provider.get_vector_size()
        vectors = collection_info.config.params.vectors

        if isinstance(vectors, dict):
            params = vectors.get(expected_name)
            if params is not None and params.size == expected_size:
                return
            names = ", ".join(
                f"'{name}' (size {p.size})" for name, p in vectors.items()
            )
            found = f"named vectors {names}" if names else "no dense vectors"
        elif vectors is not None:
            found = f"a single unnamed vector (size {vectors.size})"
        else:
            found = "no dense vectors"

        raise IncompatibleCollectionError(
            f"Collection '{collection_name}' does not match the embedding model this "
            f"server uses. The server needs a vector named '{expected_name}' with size "
            f"{expected_size}, but the collection has {found}. The vector name and size "
            "come from EMBEDDING_MODEL: use the model the collection was created with, "
            "or use a different collection."
        )

    async def _ensure_collection_exists(self, collection_name: str):
        """
        Ensure that the collection exists, creating it if necessary.

        Safe against concurrent callers: the check-then-create window is not
        atomic against the server, so two callers can both see the collection
        as missing and both attempt to create it. The loser gets a "collection
        already exists" error, which we treat as success because the
        collection now exists.

        :param collection_name: The name of the collection to ensure exists.
        """
        collection_info = await self._get_collection(collection_name)
        if collection_info is not None:
            self._check_vectors(collection_name, collection_info)
            return

        vector_size = self._embedding_provider.get_vector_size()
        vector_name = self._embedding_provider.get_vector_name()

        try:
            await self._client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    vector_name: models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                    )
                },
            )
        except ValueError as e:
            # Local mode raises ValueError("Collection ... already exists")
            # when another caller wins the race between us. Treat as success.
            if "already exists" not in str(e).lower():
                raise
        except UnexpectedResponse as e:
            # Remote mode: server returns 409 Conflict on duplicate create.
            if e.status_code != 409:
                raise

        # Creating a payload index that already exists succeeds, so callers
        # that lost the race above can safely repeat this.
        if self._field_indexes:
            for field_name, field_type in self._field_indexes.items():
                await self._client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=field_type,
                )
