import asyncio
import importlib.metadata
from typing import Optional

import click
import mcp
import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions

from .embeddings.factory import create_embedding_provider
from .qdrant import QdrantConnector


def get_package_version() -> str:
    """Get the package version using importlib.metadata."""
    try:
        return importlib.metadata.version("mcp-server-qdrant")
    except importlib.metadata.PackageNotFoundError:
        # Fall back to a default version if package is not installed
        return "0.0.0"


def serve(
    qdrant_connector: QdrantConnector,
) -> Server:
    """
    Instantiate the server and configure tools to store and find memories in Qdrant.
    :param qdrant_connector: An instance of QdrantConnector to use for storing and retrieving memories.
    """
    server = Server("qdrant")

    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        """
        Return the list of tools that the server provides. By default, there are two
        tools: one to store memories and another to find them. Finding the memories is not
        implemented as a resource, as it requires a query to be passed and resources point
        to a very specific piece of data.
        """
        return [
            types.Tool(
                name="qdrant-store-memory",
                description=(
                    "Keep the memory for later use, when you are asked to remember something."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "information": {
                            "type": "string",
                        },
                    },
                    "required": ["information"],
                },
            ),
            types.Tool(
                name="qdrant-find-memories",
                description=(
                    "Look up memories in Qdrant. Use this tool when you need to: \n"
                    " - Find memories by their content \n"
                    " - Access memories for further analysis \n"
                    " - Get some personal information about the user"
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query to search for",
                        }
                    },
                    "required": ["query"],
                },
            ),
        ]

    @server.call_tool()
    async def handle_tool_call(
        name: str, arguments: dict | None
    ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        if name not in ["qdrant-store-memory", "qdrant-find-memories"]:
            raise ValueError(f"Unknown tool: {name}")

        if name == "qdrant-store-memory":
            if not arguments or "information" not in arguments:
                raise ValueError("Missing required argument 'information'")
            information = arguments["information"]
            await qdrant_connector.store_memory(information)
            return [types.TextContent(type="text", text=f"Remembered: {information}")]

        if name == "qdrant-find-memories":
            if not arguments or "query" not in arguments:
                raise ValueError("Missing required argument 'query'")
            query = arguments["query"]
            memories = await qdrant_connector.find_memories(query)
            content = [
                types.TextContent(
                    type="text", text=f"Memories for the query '{query}'"
                ),
            ]
            for memory in memories:
                content.append(
                    types.TextContent(type="text", text=f"<memory>{memory}</memory>")
                )
            return content

        raise ValueError(f"Unknown tool: {name}")

    return server


@click.command()
@click.option(
    "--qdrant-url",
    envvar="QDRANT_URL",
    required=False,
    help="Qdrant URL",
)
@click.option(
    "--qdrant-api-key",
    envvar="QDRANT_API_KEY",
    required=False,
    help="Qdrant API key",
)
@click.option(
    "--collection-name",
    envvar="COLLECTION_NAME",
    required=True,
    help="Collection name",
)
@click.option(
    "--fastembed-model-name",
    envvar="FASTEMBED_MODEL_NAME",
    required=False,
    help="FastEmbed model name",
    default="sentence-transformers/all-MiniLM-L6-v2",
)
@click.option(
    "--embedding-provider",
    envvar="EMBEDDING_PROVIDER",
    required=False,
    help="Embedding provider to use",
    default="fastembed",
    type=click.Choice(["fastembed"], case_sensitive=False),
)
@click.option(
    "--embedding-model",
    envvar="EMBEDDING_MODEL",
    required=False,
    help="Embedding model name",
    default="sentence-transformers/all-MiniLM-L6-v2",
)
@click.option(
    "--qdrant-local-path",
    envvar="QDRANT_LOCAL_PATH",
    required=False,
    help="Qdrant local path",
)
def main(
    qdrant_url: Optional[str],
    qdrant_api_key: str,
    collection_name: Optional[str],
    fastembed_model_name: Optional[str],
    embedding_provider: str,
    embedding_model: str,
    qdrant_local_path: Optional[str],
):
    # XOR of url and local path, since we accept only one of them
    if not (bool(qdrant_url) ^ bool(qdrant_local_path)):
        raise ValueError(
            "Exactly one of qdrant-url or qdrant-local-path must be provided"
        )

    # Warn if fastembed_model_name is provided, as this is going to be deprecated
    if fastembed_model_name:
        click.echo(
            "Warning: --fastembed-model-name parameter is deprecated and will be removed in a future version. "
            "Please use --embedding-provider and --embedding-model instead",
            err=True,
        )

    async def _run():
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            # Create the embedding provider
            provider = create_embedding_provider(
                provider_type=embedding_provider, model_name=embedding_model
            )

            # Create the Qdrant connector
            qdrant_connector = QdrantConnector(
                qdrant_url,
                qdrant_api_key,
                collection_name,
                provider,
                qdrant_local_path,
            )

            # Create and run the server
            server = serve(qdrant_connector)
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="qdrant",
                    server_version=get_package_version(),
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )

    asyncio.run(_run())
