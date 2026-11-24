import json
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from typing import Annotated, Any, Optional
from xml.sax.saxutils import escape

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.utilities.logging import get_logger
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

# A child of FastMCP's logger, so it shares its handler and FASTMCP_LOG_LEVEL.
logger = get_logger(__name__)


def _server_version() -> str | None:
    """Version of this package, reported to MCP clients in `serverInfo`."""
    try:
        return package_version("mcp-server-qdrant")
    except PackageNotFoundError:
        # Running from source without installing: let FastMCP use its default.
        return None


def _error_types(error: BaseException) -> str:
    """Name the exception types in the cause chain, up to the first one with a message.

    Some errors carry no message at all, such as `httpx.ReadTimeout("")` when Qdrant
    does not answer in time, so the client would otherwise see an empty error.
    """
    names: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        if names and str(current):
            break
        seen.add(id(current))
        if not names or names[-1] != type(current).__name__:
            names.append(type(current).__name__)
        current = current.__cause__ or current.__context__
    return " caused by ".join(names)


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

        # Without this, FastMCP reports its own version as the server version.
        settings.setdefault("version", _server_version())
        super().__init__(name=name, instructions=instructions, **settings)

        self.setup_tools()

    def format_entry(self, entry: Entry) -> str:
        """
        Feel free to override this method in your subclass to customize the format of the entry.

        Content and metadata are XML-escaped so stored values cannot inject the
        formatter's own tags (``</content>``, ``<metadata>`` and so on).
        """
        entry_metadata = json.dumps(entry.metadata) if entry.metadata else ""
        return (
            f"<entry><content>{escape(entry.content)}</content>"
            f"<metadata>{escape(entry_metadata)}</metadata></entry>"
        )

    def format_skipped_note(self, skipped: int) -> str:
        """
        Tell the model that some matches were left out, so it does not treat
        the results as complete.
        """
        return (
            f"Note: {skipped} matching point(s) were left out because their payloads "
            "have no 'document' field. This tool only returns points that store "
            "their text in that field."
        )

    def setup_tools(self):
        """
        Register the tools in the server.
        """

        async def store(
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
            :param information: The information to store.
            :param metadata: JSON metadata to store with the information, optional.
            :param collection_name: The name of the collection to store the information in, optional. If not provided,
                                    the default collection is used.
            :return: A message indicating that the information was stored.
            """
            logger.debug("Storing information %s in Qdrant", information)

            entry = Entry(content=information, metadata=metadata)

            try:
                await self.qdrant_connector.store(
                    entry, collection_name=collection_name
                )
            except Exception as error:
                if str(error):
                    raise
                raise ToolError(
                    f"Error calling tool 'qdrant-store': {_error_types(error)}"
                ) from error
            if collection_name:
                return f"Remembered: {information} in collection {collection_name}"
            return f"Remembered: {information}"

        async def find(
            query: Annotated[str, Field(description="What to search for")],
            collection_name: Annotated[
                str, Field(description="The collection to search in")
            ],
            query_filter: ArbitraryFilter | None = None,
        ) -> list[str] | None:
            """
            Find memories in Qdrant.
            :param query: The query to use for the search.
            :param collection_name: The name of the collection to search in, optional. If not provided,
                                    the default collection is used.
            :param query_filter: The filter to apply to the query.
            :return: A list of entries found or None.
            """

            logger.debug("Query filter: %s", query_filter)

            query_filter = models.Filter(**query_filter) if query_filter else None

            logger.debug("Finding results for query %s", query)

            try:
                entries = await self.qdrant_connector.search(
                    query,
                    collection_name=collection_name,
                    limit=self.qdrant_settings.search_limit,
                    query_filter=query_filter,
                )
            except Exception as error:
                if str(error):
                    raise
                raise ToolError(
                    f"Error calling tool 'qdrant-find': {_error_types(error)}"
                ) from error
            if not entries and entries.skipped == 0:
                return None
            content = [
                f"Results for the query '{query}'",
            ]
            for entry in entries:
                content.append(self.format_entry(entry))
            if entries.skipped > 0:
                content.append(self.format_skipped_note(entries.skipped))
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

        self.tool(
            find_foo,
            name="qdrant-find",
            description=self.tool_settings.tool_find_description,
        )

        if not self.qdrant_settings.read_only:
            # Those methods can modify the database
            self.tool(
                store_foo,
                name="qdrant-store",
                description=self.tool_settings.tool_store_description,
            )
