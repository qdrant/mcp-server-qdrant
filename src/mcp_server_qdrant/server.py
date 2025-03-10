import logging
from contextlib import asynccontextmanager
from typing import Annotated, AsyncIterator, List, Optional

import typer
from mcp.server import Server
from mcp.server.fastmcp import Context, FastMCP

from mcp_server_qdrant.embeddings.factory import (
    EmbeddingProviderType,
    create_embedding_provider,
)
from mcp_server_qdrant.qdrant import QdrantConnector
from mcp_server_qdrant.settings import EmbeddingProviderSettings, QdrantSettings

logger = logging.getLogger(__name__)

app = typer.Typer(name="mcp-server-qdrant")


def create_mcp(
    collection_name: Annotated[
        str,
        typer.Option(
            "--collection-name",
            envvar="COLLECTION_NAME",
        ),
    ],
    qdrant_url: Annotated[
        Optional[str],
        typer.Option(
            "--qdrant-url",
            show_default=True,
            envvar="QDRANT_URL",
        ),
    ] = None,
    qdrant_api_key: Annotated[
        Optional[str],
        typer.Option(
            "--qdrant-api-key",
            envvar="QDRANT_API_KEY",
        ),
    ] = None,
    embedding_provider_type: Annotated[
        EmbeddingProviderType,
        typer.Option(
            "--embedding-provider",
            envvar="EMBEDDING_PROVIDER_TYPE",
        ),
    ] = EmbeddingProviderType.FASTEMBED,
    embedding_model_name: Annotated[
        str,
        typer.Option(
            "--embedding-model",
            envvar="EMBEDDING_MODEL_NAME",
        ),
    ] = "sentence-transformers/all-MiniLM-L6-v2",
    qdrant_local_path: Annotated[
        Optional[str], typer.Option(envvar="QDRANT_LOCAL_PATH")
    ] = None,
):
    """
    Create the FastMCP server using the CLI parameters and declare
    all the components available for MCP clients.

    One of the following options must be specified:

    - --qdrant-url: The URL of the Qdrant instance.

    - --qdrant-local-path: The local path to the Qdrant instance.
    """

    if qdrant_url and qdrant_local_path:
        raise ValueError(
            "Only one of --qdrant-url or --qdrant-local-path can be specified"
        )

    @asynccontextmanager
    async def server_lifespan(server: Server) -> AsyncIterator[dict]:  # noqa
        """
        Context manager to handle the lifespan of the server.
        This is used to configure the embedding provider and Qdrant connector.
        """
        try:
            # Embedding provider is created with a factory function so we can add
            # some more providers in the future. Currently, only FastEmbed is supported.
            embedding_provider_settings = EmbeddingProviderSettings(
                provider_type=embedding_provider_type,
                model_name=embedding_model_name,
            )
            embedding_provider = create_embedding_provider(embedding_provider_settings)
            logger.info(
                f"Using embedding provider {embedding_provider_settings.provider_type} with "
                f"model {embedding_provider_settings.model_name}"
            )

            qdrant_configuration = QdrantSettings(
                location=qdrant_url,
                api_key=qdrant_api_key,
                collection_name=collection_name,
                local_path=qdrant_local_path,
            )
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


def main():
    """
    An entrypoint for the mcp-server-qdrant script defined in pyproject.toml.
    """
    typer.run(create_mcp)


if __name__ == "__main__":
    main()
