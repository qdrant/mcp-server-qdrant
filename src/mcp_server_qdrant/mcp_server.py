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
        entry_id = f"<id>{entry.id}</id>" if entry.id else ""
        return f"<entry>{entry_id}<content>{entry.content}</content><metadata>{entry_metadata}</metadata></entry>"

    def setup_tools(self):
        """
        Register the tools in the server.
        """

        async def store(
            ctx: Context,
            id: Annotated[
                str | None,
                Field(description="Point ID. If omitted, a new point is created."),
            ],
            information: Annotated[str, Field(description="Text to store")],
            collection_name: Annotated[
                str, Field(description="The collection to store the information in")
            ],
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

            entry = Entry(content=information, metadata=metadata, id=id)

            await self.qdrant_connector.store(entry, collection_name=collection_name)
            if collection_name:
                return f"Remembered: {information} in collection {collection_name}"
            return f"Remembered: {information}"

        async def find(
            ctx: Context,
            query: Annotated[str, Field(description="What to search for")],
            collection_name: Annotated[
                str, Field(description="The collection to search in")
            ],
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

            parsed_query_filter = (
                models.Filter(**query_filter) if query_filter else None
            )

            await ctx.debug(f"Finding results for query {query}")

            entries = await self.qdrant_connector.search(
                query,
                collection_name=collection_name,
                limit=self.qdrant_settings.search_limit,
                query_filter=parsed_query_filter,
            )
            if not entries:
                return None
            content = [
                f"Results for the query '{query}'",
            ]
            for entry in entries:
                content.append(self.format_entry(entry))
            return content

        find_foo = find
        store_foo = store

        filterable_conditions = (
            self.qdrant_settings.filterable_fields_dict_with_conditions()
        )

        if len(filterable_conditions) > 0:
            find_foo = wrap_filters(find_foo, filterable_conditions)
        elif not self.qdrant_settings.allow_arbitrary_filter:
            find_foo = make_partial_function(find_foo, {"query_filter": None})

        if self.qdrant_settings.collection_name:
            find_foo = make_partial_function(
                find_foo, {"collection_name": self.qdrant_settings.collection_name}
            )
            store_foo = make_partial_function(
                store_foo, {"collection_name": self.qdrant_settings.collection_name}
            )

        # Add new tools for point operations
        async def get_point(
            ctx: Context,
            point_id: Annotated[
                str, Field(description="The ID of the point to retrieve")
            ],
            collection_name: Annotated[
                str, Field(description="The collection to get the point from")
            ],
        ) -> str:
            """
            Get a specific point by its ID.
            :param ctx: The context for the request.
            :param point_id: The ID of the point to retrieve.
            :param collection_name: The name of the collection to get the point from.
            :return: The point information or error message.
            """
            await ctx.debug(
                f"Getting point {point_id} from collection {collection_name}"
            )

            entry = await self.qdrant_connector.get_point_by_id(
                point_id, collection_name=collection_name
            )

            if entry:
                return self.format_entry(entry)
            else:
                return f"Point with ID {point_id} not found in collection {collection_name}"

        async def delete_point(
            ctx: Context,
            point_id: Annotated[
                str, Field(description="The ID of the point to delete")
            ],
            collection_name: Annotated[
                str, Field(description="The collection to delete the point from")
            ],
        ) -> str:
            """
            Delete a specific point by its ID.
            :param ctx: The context for the request.
            :param point_id: The ID of the point to delete.
            :param collection_name: The name of the collection to delete the point from.
            :return: Success or error message.
            """
            await ctx.debug(
                f"Deleting point {point_id} from collection {collection_name}"
            )

            success = await self.qdrant_connector.delete_point_by_id(
                point_id, collection_name=collection_name
            )

            if success:
                return f"Successfully deleted point {point_id} from collection {collection_name}"
            else:
                return f"Failed to delete point {point_id} - point not found or collection doesn't exist"

        async def update_point_payload(
            ctx: Context,
            point_id: Annotated[
                str, Field(description="The ID of the point to update")
            ],
            collection_name: Annotated[
                str, Field(description="The collection containing the point")
            ],
            metadata: Annotated[
                Metadata,
                Field(
                    description="New metadata to set for the point. Any json is accepted."
                ),
            ],
        ) -> str:
            """
            Update the payload (metadata) of a specific point by its ID.
            :param ctx: The context for the request.
            :param point_id: The ID of the point to update.
            :param collection_name: The name of the collection containing the point.
            :param metadata: New metadata to set for the point.
            :return: Success or error message.
            """
            await ctx.debug(
                f"Updating payload for point {point_id} in collection {collection_name}"
            )

            success = await self.qdrant_connector.update_point_payload(
                point_id, metadata, collection_name=collection_name
            )

            if success:
                return f"Successfully updated payload for point {point_id} in collection {collection_name}"
            else:
                return f"Failed to update payload for point {point_id} - point not found or collection doesn't exist"

        # Add collection management tools
        async def get_collections(ctx: Context) -> list[str]:
            """
            Get a list of all collections in the Qdrant server.
            :param ctx: The context for the request.
            :return: A list of collection names.
            """
            await ctx.debug("Getting list of collections")
            collection_names = await self.qdrant_connector.get_collection_names()
            return collection_names

        async def get_collection_details(
            ctx: Context,
            collection_name: Annotated[
                str, Field(description="The name of the collection to get details for")
            ],
        ) -> str:
            """
            Get detailed information about a specific collection.
            :param ctx: The context for the request.
            :param collection_name: The name of the collection to get details for.
            :return: Detailed information about the collection.
            """
            await ctx.debug(f"Getting details for collection {collection_name}")
            try:
                details = await self.qdrant_connector.get_collection_details(
                    collection_name
                )
                return json.dumps(details, indent=2)
            except Exception as e:
                return f"Error getting collection details: {str(e)}"

        # Apply collection name defaults to new tools
        get_point_foo = get_point
        delete_point_foo = delete_point
        update_point_payload_foo = update_point_payload

        if self.qdrant_settings.collection_name:
            get_point_foo = make_partial_function(
                get_point_foo, {"collection_name": self.qdrant_settings.collection_name}
            )
            delete_point_foo = make_partial_function(
                delete_point_foo,
                {"collection_name": self.qdrant_settings.collection_name},
            )
            update_point_payload_foo = make_partial_function(
                update_point_payload_foo,
                {"collection_name": self.qdrant_settings.collection_name},
            )

        self.tool(
            find_foo,
            name="qdrant-find",
            description=self.tool_settings.tool_find_description,
        )

        self.tool(
            get_point_foo,
            name="qdrant-get-point",
            description="Get a specific point by its ID from a Qdrant collection.",
        )

        self.tool(
            get_collections,
            name="qdrant-get-collections",
            description="Get a list of all collections in the Qdrant server.",
        )

        self.tool(
            get_collection_details,
            name="qdrant-get-collection-details",
            description="Get detailed information about a specific collection including status, configuration, and statistics.",
        )

        if not self.qdrant_settings.read_only:
            # Those methods can modify the database
            self.tool(
                store_foo,
                name="qdrant-store",
                description=self.tool_settings.tool_store_description,
            )

            self.tool(
                delete_point_foo,
                name="qdrant-delete-point",
                description="Delete a specific point by its ID from a Qdrant collection.",
            )

            self.tool(
                update_point_payload_foo,
                name="qdrant-update-point-payload",
                description="Update the payload (metadata) of a specific point by its ID.",
            )
