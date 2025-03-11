import json
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, List

from mcp.server import Server
from mcp.server.fastmcp import Context, FastMCP

from mcp_server_qdrant.embeddings.factory import create_embedding_provider
from mcp_server_qdrant.qdrant import Entry, Metadata, QdrantConnector
from mcp_server_qdrant.settings import (
    EmbeddingProviderSettings,
    QdrantSettings,
    ToolSettings,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def server_lifespan(server: Server) -> AsyncIterator[dict]:  # noqa
    """
    Context manager to handle the lifespan of the server.
    This is used to configure the embedding provider and Qdrant connector.
    All the configuration is now loaded from the environment variables.
    Settings handle that for us.
    """
    try:
        # Embedding provider is created with a factory function so we can add
        # some more providers in the future. Currently, only FastEmbed is supported.
        embedding_provider_settings = EmbeddingProviderSettings()
        embedding_provider = create_embedding_provider(embedding_provider_settings)
        logger.info(
            f"Using embedding provider {embedding_provider_settings.provider_type} with "
            f"model {embedding_provider_settings.model_name}"
        )

        qdrant_configuration = QdrantSettings()
        qdrant_connector = QdrantConnector(
            qdrant_configuration.location,
            qdrant_configuration.api_key,
            qdrant_configuration.collection_name,
            embedding_provider,
            qdrant_configuration.local_path,
        )
        logger.info(
            f"Connecting to Qdrant at {qdrant_configuration.get_qdrant_location()}"
        )

        yield {
            "embedding_provider": embedding_provider,
            "qdrant_connector": qdrant_connector,
        }
    except Exception as e:
        logger.error(e)
        raise e
    finally:
        pass


# FastMCP is an alternative interface for declaring the capabilities
# of the server. Its API is based on FastAPI.
mcp = FastMCP("mcp-server-qdrant", lifespan=server_lifespan)

# Load the tool settings from the env variables, if they are set,
# or use the default values otherwise.
tool_settings = ToolSettings()


@mcp.tool(name="qdrant-store", description=tool_settings.tool_store_description)
async def store(
    ctx: Context,
    information: str,
    # The `metadata` parameter is defined as non-optional, but it can be None.
    # If we set it to be optional, some of the MCP clients, like Cursor, cannot
    # handle the optional parameter correctly.
    metadata: Metadata = None,
) -> str:
    """
    Store some information in Qdrant.
    :param ctx: The context for the request.
    :param information: The information to store.
    :param metadata: JSON metadata to store with the information, optional.
    :return: A message indicating that the information was stored.
    """
    await ctx.debug(f"Storing information {information} in Qdrant")
    qdrant_connector: QdrantConnector = ctx.request_context.lifespan_context[
        "qdrant_connector"
    ]
    entry = Entry(content=information, metadata=metadata)
    await qdrant_connector.store(entry)
    return f"Remembered: {information}"


@mcp.tool(name="qdrant-find", description=tool_settings.tool_find_description)
async def find(ctx: Context, query: str) -> List[str]:
    """
    Find memories in Qdrant.
    :param ctx: The context for the request.
    :param query: The query to use for the search.
    :return: A list of entries found.
    """
    await ctx.debug(f"Finding results for query {query}")
    qdrant_connector: QdrantConnector = ctx.request_context.lifespan_context[
        "qdrant_connector"
    ]
    entries = await qdrant_connector.search(query)
    if not entries:
        return [f"No information found for the query '{query}'"]
    content = [
        f"Results for the query '{query}'",
    ]
    for entry in entries:
        # Format the metadata as a JSON string and produce XML-like output
        entry_metadata = json.dumps(entry.metadata) if entry.metadata else ""
        content.append(
            f"<entry><content>{entry.content}</content><metadata>{entry_metadata}</metadata></entry>"
        )
    return content
