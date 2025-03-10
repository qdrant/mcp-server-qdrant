import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, List

from mcp.server import Server
from mcp.server.fastmcp import Context, FastMCP

from mcp_server_qdrant.embeddings.factory import create_embedding_provider
from mcp_server_qdrant.qdrant import QdrantConnector
from mcp_server_qdrant.settings import EmbeddingProviderSettings, QdrantSettings

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


mcp = FastMCP("mcp-server-qdrant", lifespan=server_lifespan)


@mcp.tool(
    name="qdrant-store-memory",
    description=(
        "Keep the memory for later use, when you are asked to remember something."
    ),
)
async def store(information: str, ctx: Context) -> str:
    """
    Store a memory in Qdrant.
    :param information: The information to store.
    :param ctx: The context for the request.
    :return: A message indicating that the information was stored.
    """
    await ctx.debug(f"Storing information {information} in Qdrant")
    qdrant_connector: QdrantConnector = ctx.request_context.lifespan_context[
        "qdrant_connector"
    ]
    await qdrant_connector.store(information)
    return f"Remembered: {information}"


@mcp.tool(
    name="qdrant-find-memories",
    description=(
        "Look up memories in Qdrant. Use this tool when you need to: \n"
        " - Find memories by their content \n"
        " - Access memories for further analysis \n"
        " - Get some personal information about the user"
    ),
)
async def find(query: str, ctx: Context) -> List[str]:
    """
    Find memories in Qdrant.
    :param query: The query to use for the search.
    :param ctx: The context for the request.
    :return: A list of entries found.
    """
    await ctx.debug(f"Finding points for query {query}")
    qdrant_connector: QdrantConnector = ctx.request_context.lifespan_context[
        "qdrant_connector"
    ]
    entries = await qdrant_connector.search(query)
    if not entries:
        return [f"No memories found for the query '{query}'"]
    content = [
        f"Memories for the query '{query}'",
    ]
    for entry in entries:
        content.append(f"<entry>{entry}</entry>")
    return content
