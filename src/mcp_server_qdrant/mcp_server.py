import json
import logging
from typing import Annotated, Any, Optional

from fastmcp import Context, FastMCP
from pydantic import Field
from qdrant_client import models

from mcp_server_qdrant.common.filters import make_indexes
from mcp_server_qdrant.common.func_tools import make_partial_function
from mcp_server_qdrant.common.wrap_filters import wrap_filters
from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.embeddings.factory import create_embedding_provider
from mcp_server_qdrant.qdrant import ArbitraryFilter, Entry, Metadata, QdrantConnector
from mcp_server_qdrant.settings import (
    EmbeddingProviderSettings,
    QdrantSettings,
    ToolSettings,
)

logger = logging.getLogger(__name__)


# FastMCP is an alternative interface for declaring the capabilities
# of the server. Its API is based on FastAPI.
class QdrantMCPServer(FastMCP):
    """
    A MCP server for Qdrant.
    """

    def __init__(
        self,
        tool_settings: ToolSettings,
        qdrant_settings: QdrantSettings,
        embedding_provider_settings: Optional[EmbeddingProviderSettings] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        name: str = "mcp-server-qdrant",
        instructions: str | None = None,
        **settings: Any,
    ):
        self.tool_settings = tool_settings
        self.qdrant_settings = qdrant_settings

        if embedding_provider_settings and embedding_provider:
            raise ValueError(
                "Cannot provide both embedding_provider_settings and embedding_provider"
            )

        if not embedding_provider_settings and not embedding_provider:
            raise ValueError(
                "Must provide either embedding_provider_settings or embedding_provider"
            )

        self.embedding_provider_settings: Optional[EmbeddingProviderSettings] = None
        self.embedding_provider: Optional[EmbeddingProvider] = None

        if embedding_provider_settings:
            self.embedding_provider_settings = embedding_provider_settings
            self.embedding_provider = create_embedding_provider(
                embedding_provider_settings
            )
        else:
            self.embedding_provider_settings = None
            self.embedding_provider = embedding_provider

        assert self.embedding_provider is not None, "Embedding provider is required"

        self.qdrant_connector = QdrantConnector(
            qdrant_settings.location,
            qdrant_settings.api_key,
            qdrant_settings.collection_name,
            self.embedding_provider,
            qdrant_settings.local_path,
            make_indexes(qdrant_settings.filterable_fields_dict()),
        )

        super().__init__(name=name, instructions=instructions, **settings)

        self.setup_tools()

    def format_entry(self, entry: Entry) -> str:
        """
        Feel free to override this method in your subclass to customize the format of the entry.
        """
        entry_metadata = json.dumps(entry.metadata) if entry.metadata else ""
        return f"<entry><content>{entry.content}</content><metadata>{entry_metadata}</metadata></entry>"

    def setup_tools(self):
        """
        Register the tools in the server.
        """

        async def store(
            ctx: Context,
            information: Annotated[str, Field(description="Text to store")],
            collection_name: Annotated[
                str | None,
                Field(description="The collection to store the information in"),
            ] = None,
            # The `metadata` parameter is defined as non-optional, but it can be None.
            # If we set it to be optional, some of the MCP clients, like Cursor, cannot
            # handle the optional parameter correctly.
            metadata: Annotated[
                Metadata | None,
                Field(
                    description="Extra metadata stored along with memorised information. Any json is accepted."
                ),
            ] = None,
        ) -> str:
            """
            Store some information in Qdrant.
            :param ctx: The context for the request.
            :param information: The information to store.
            :param metadata: JSON metadata to store with the information, optional.
            :param collection_name: The name of the collection to store the information in, optional. If not provided,
                                    the default collection is used.
            :return: A message indicating that the information was stored.
            """
            await ctx.debug(f"Storing information {information} in Qdrant")

            # Use default collection if none specified
            effective_collection = (
                collection_name or self.qdrant_settings.collection_name
            )

            entry = Entry(content=information, metadata=metadata)

            await self.qdrant_connector.store(entry, collection_name=effective_collection)
            if effective_collection:
                return f"Remembered: {information} in collection {effective_collection}"
            return f"Remembered: {information}"

        async def find(
            ctx: Context,
            query: Annotated[str, Field(description="What to search for")],
            collection_name: Annotated[
                str | None, Field(description="The collection to search in")
            ] = None,
            query_filter: ArbitraryFilter | None = None,
        ) -> list[str] | None:
            """
            Find memories in Qdrant.
            :param ctx: The context for the request.
            :param query: The query to use for the search.
            :param collection_name: The name of the collection to search in, optional. If not provided,
                                    the default collection is used.
            :param query_filter: The filter to apply to the query.
            :return: A list of entries found or None.
            """

            # Log query_filter
            await ctx.debug(f"Query filter: {query_filter}")

            query_filter = models.Filter(**query_filter) if query_filter else None

            await ctx.debug(f"Finding results for query {query}")

            # Use default collection if none specified
            effective_collection = (
                collection_name or self.qdrant_settings.collection_name
            )

            entries = await self.qdrant_connector.search(
                query,
                collection_name=effective_collection,
                limit=self.qdrant_settings.search_limit,
                query_filter=query_filter,
            )
            if not entries:
                return None
            content = [
                f"Results for the query '{query}'",
            ]
            for entry in entries:
                content.append(self.format_entry(entry))
            return content

        async def hybrid_find(
            ctx: Context,
            query: Annotated[str, Field(description="What to search for")],
            collection_name: Annotated[
                str | None, Field(description="The collection to search in")
            ] = None,
            fusion_method: Annotated[
                str,
                Field(
                    description="Fusion method to use: 'rrf' (Reciprocal Rank Fusion) or 'dbsf' (Distribution-Based Score Fusion)"
                ),
            ] = "rrf",
            dense_limit: Annotated[
                int, Field(description="Limit for dense vector prefetch")
            ] = 10,
            sparse_limit: Annotated[
                int, Field(description="Limit for sparse vector prefetch")
            ] = 10,
            final_limit: Annotated[
                int, Field(description="Final limit after fusion")
            ] = 10,
            query_filter: ArbitraryFilter | None = None,
        ) -> list[str] | None:
            """
            Find memories in Qdrant using hybrid search (dense + sparse vectors).
            :param ctx: The context for the request.
            :param query: The query to use for the search.
            :param collection_name: The name of the collection to search in, optional.
            :param fusion_method: The fusion method to use ('rrf' or 'dbsf').
            :param dense_limit: Limit for dense vector prefetch.
            :param sparse_limit: Limit for sparse vector prefetch.
            :param final_limit: Final limit after fusion.
            :param query_filter: The filter to apply to the query.
            :return: A list of entries found or None.
            """
            await ctx.debug(f"Query filter: {query_filter}")
            query_filter = models.Filter(**query_filter) if query_filter else None

            await ctx.debug(
                f"Hybrid search for query {query} with fusion method {fusion_method}"
            )

            # Use default collection if none specified
            effective_collection = (
                collection_name or self.qdrant_settings.collection_name
            )

            entries = await self.qdrant_connector.find_hybrid(
                query,
                collection_name=effective_collection,
                fusion_method=fusion_method,
                dense_limit=dense_limit,
                sparse_limit=sparse_limit,
                final_limit=final_limit,
                query_filter=query_filter,
            )

            if not entries:
                return None

            content = [f"Hybrid search results for the query '{query}'"]
            for entry in entries:
                content.append(self.format_entry(entry))
            return content

        find_foo = find
        store_foo = store
        hybrid_find_foo = hybrid_find

        filterable_conditions = (
            self.qdrant_settings.filterable_fields_dict_with_conditions()
        )

        if len(filterable_conditions) > 0:
            find_foo = wrap_filters(find_foo, filterable_conditions)
            hybrid_find_foo = wrap_filters(hybrid_find_foo, filterable_conditions)
        elif not self.qdrant_settings.allow_arbitrary_filter:
            find_foo = make_partial_function(find_foo, {"query_filter": None})
            hybrid_find_foo = make_partial_function(
                hybrid_find_foo, {"query_filter": None}
            )

        self.tool(
            find_foo,
            name="qdrant-find",
            description=self.tool_settings.tool_find_description,
        )

        self.tool(
            hybrid_find_foo,
            name="qdrant-hybrid-find",
            description=self.tool_settings.tool_hybrid_find_description,
        )

        async def get_point(
            ctx: Context,
            point_id: Annotated[str, Field(description="The ID of the point to retrieve")],
            collection_name: Annotated[
                str | None, Field(description="The collection to retrieve from")
            ] = None,
        ) -> str | None:
            """
            Get a specific point by ID from a collection.
            """
            effective_collection = (
                collection_name or self.qdrant_settings.collection_name
            )
            entry = await self.qdrant_connector.get_point(
                point_id, collection_name=effective_collection
            )
            if entry:
                return self.format_entry(entry)
            return None

        async def delete_point(
            ctx: Context,
            point_id: Annotated[str, Field(description="The ID of the point to delete")],
            collection_name: Annotated[
                str | None, Field(description="The collection to delete from")
            ] = None,
        ) -> str:
            """
            Delete a specific point by ID from a collection.
            """
            effective_collection = (
                collection_name or self.qdrant_settings.collection_name
            )
            success = await self.qdrant_connector.delete_point(
                point_id, collection_name=effective_collection
            )
            if success:
                return f"Point {point_id} deleted successfully from collection {effective_collection}"
            return f"Failed to delete point {point_id}"

        async def update_point_payload(
            ctx: Context,
            point_id: Annotated[str, Field(description="The ID of the point to update")],
            metadata: Annotated[
                Metadata,
                Field(description="The new metadata to set for the point"),
            ],
            collection_name: Annotated[
                str | None, Field(description="The collection containing the point")
            ] = None,
        ) -> str:
            """
            Update the metadata/payload for a specific point.
            """
            effective_collection = (
                collection_name or self.qdrant_settings.collection_name
            )
            success = await self.qdrant_connector.update_point_payload(
                point_id, metadata, collection_name=effective_collection
            )
            if success:
                return f"Point {point_id} metadata updated successfully"
            return f"Failed to update point {point_id} metadata"

        async def get_collections(ctx: Context) -> list[str]:
            """
            List all available collections in Qdrant.
            """
            return await self.qdrant_connector.get_collection_names()

        async def get_collection_details(
            ctx: Context,
            collection_name: Annotated[
                str, Field(description="The name of the collection to get details for")
            ],
        ) -> str:
            """
            Get detailed information about a specific collection.
            """
            import json

            info = await self.qdrant_connector.get_collection_info(collection_name)
            return json.dumps(info, indent=2)

        self.tool(
            get_point,
            name="qdrant-get-point",
            description="Retrieve a specific point by ID from a collection",
        )

        self.tool(
            get_collections,
            name="qdrant-get-collections",
            description="List all available collections in Qdrant",
        )

        self.tool(
            get_collection_details,
            name="qdrant-get-collection-details",
            description="Get detailed information about a specific collection including status, vector count, and configuration",
        )

        if not self.qdrant_settings.read_only:
            # Those methods can modify the database
            self.tool(
                store_foo,
                name="qdrant-store",
                description=self.tool_settings.tool_store_description,
            )

            self.tool(
                delete_point,
                name="qdrant-delete-point",
                description="Delete a specific point by ID from a collection",
            )

            self.tool(
                update_point_payload,
                name="qdrant-update-point-payload",
                description="Update the metadata/payload for a specific point",
            )
