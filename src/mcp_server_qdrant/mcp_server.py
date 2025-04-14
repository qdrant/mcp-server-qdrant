import json
import logging
from typing import Annotated, Any, List

from mcp.server.fastmcp import Context, FastMCP
from pydantic import Field

from mcp_server_qdrant.embeddings.factory import create_embedding_provider
from mcp_server_qdrant.qdrant import Entry, Metadata, QdrantConnector
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
    An MCP server for Qdrant.
    """

    METADATA_COLLECTION_NAME = "__mcp_metadata__"

    def __init__(
        self,
        tool_settings: ToolSettings,
        qdrant_settings: QdrantSettings,
        embedding_provider_settings: EmbeddingProviderSettings,
        name: str = "mcp-server-qdrant",
        instructions: str | None = None,
        **settings: Any,
    ):
        self.tool_settings = tool_settings
        self.qdrant_settings = qdrant_settings
        self.embedding_provider_settings = embedding_provider_settings
        self.embedding_provider = create_embedding_provider(embedding_provider_settings)
        self.qdrant_connector = QdrantConnector(
            qdrant_settings.location,
            qdrant_settings.api_key,
            self.embedding_provider,
            qdrant_settings.local_path,
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
        self.add_tool(
            self.find,
            name="qdrant-find",
            description=self.tool_settings.tool_find_description,
        )
        self.add_tool(
            self.list_collections,
            name="qdrant-list-collections",
            description=self.tool_settings.tool_list_collections_description,
        )

        if not self.qdrant_settings.read_only:
            # Those methods can modify the database
            self.add_tool(
                self.store,
                name="qdrant-store",
                description=self.tool_settings.tool_store_description,
            )
            self.add_tool(
                self.create_collection,
                name="qdrant-create-collection",
                description=self.tool_settings.tool_create_collection_description,
            )

        # TODO: if the default collection does not exist, create it here

    async def find(
        self,
        ctx: Context,
        # Annotation help to describe the meaning of the parameter. MCP SDK is based on Pydantic.
        query: Annotated[
            str, Field(description="A natural language query to search for.")
        ],
        collection_name: Annotated[
            str,
            Field(
                description="Name of the collection to search in. Please list the existing collections to know which one to use."
            ),
        ],
    ) -> List[str]:
        """
        Find memories in a specific Qdrant collection.
        """
        if not collection_name and not self.qdrant_settings.collection_name:
            return ["Please specify the collection name"]

        if not collection_name:
            assert self.qdrant_settings.collection_name is not None
            collection_name = self.qdrant_settings.collection_name
            await ctx.debug(f"Using default collection {collection_name}")

        await ctx.debug(f"Finding results for query '{query}' in {collection_name}")

        entries = await self.qdrant_connector.search(
            query,
            collection_name=collection_name,
            limit=self.qdrant_settings.search_limit,
        )
        if not entries:
            return [f"No information found for the query '{query}'"]
        content = [
            f"Results for the query '{query}'",
        ]
        for entry in entries:
            content.append(self.format_entry(entry))
        return content

    async def store(
        self,
        ctx: Context,
        # Annotation help to describe the meaning of the parameter. MCP SDK is based on Pydantic.
        information: Annotated[
            str, Field(description="Natural language information to store.")
        ],
        collection_name: Annotated[
            str,
            Field(
                description=(
                    "Name of the collection to store the information in. Please list the existing collections to know "
                    "which one to use, depending on the type of information."
                )
            ),
        ],
        # The `metadata` parameter is defined as non-optional, but it can be None.
        # If we set it to be optional, some of the MCP clients, like Cursor, cannot
        # handle the optional parameter correctly.
        metadata: Annotated[
            Metadata,
            Field(description="JSON metadata to store with the information, optional."),
        ] = None,  # type: ignore
    ) -> str:
        """
        Store some information in a specific Qdrant collection. Please list the existing collections to
        know which one to use. The `information` parameter is the information to store, and `metadata` is
        optional JSON metadata to store with the information. Please extract the metadata from the information
        if possible.
        """
        if not collection_name and not self.qdrant_settings.collection_name:
            return "Please specify the collection name"

        if not collection_name:
            assert self.qdrant_settings.collection_name is not None
            collection_name = self.qdrant_settings.collection_name
            await ctx.debug(f"Using default collection {collection_name}")

        await ctx.debug(
            f"Storing information {information} in Qdrant (collection {collection_name})"
        )

        entry = Entry(content=information, metadata=metadata)
        await self.qdrant_connector.store(entry, collection_name=collection_name)
        return f"Remembered: {information} in collection {collection_name}"

    async def list_collections(self, ctx: Context) -> List[str]:
        """
        List all the collections in Qdrant, along with their purpose descriptions.
        Use this tool always when you wonder which collection to use.
        """
        await ctx.debug("Listing collections")

        content = ["Available collections:"]
        async for entry in self.qdrant_connector.iter_all(
            collection_name=self.METADATA_COLLECTION_NAME
        ):
            collection_purpose = (
                f": {entry.metadata.get('description')}" if entry.metadata else ""
            )
            collection_descriptor = f"- `{entry.content}`{collection_purpose}"
            content.append(collection_descriptor)

        if len(content) == 1:
            content.append("No collections found")

        return content

    async def create_collection(
        self,
        ctx: Context,
        # Annotation help to describe the meaning of the parameter. MCP SDK is based on Pydantic.
        collection_name: Annotated[
            str, Field(description="Name of the collection to create.")
        ],
        description: Annotated[
            str, Field(description="Purpose description of the collection.")
        ],
    ) -> str:
        """
        Create a new collection in Qdrant with the specified name and purpose description.
        """
        await ctx.debug(f"Creating collection {collection_name}")

        await self.qdrant_connector.store(
            Entry(content=collection_name, metadata={"description": description}),
            collection_name=self.METADATA_COLLECTION_NAME,
        )

        return f"Created collection {collection_name}"
