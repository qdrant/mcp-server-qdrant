import json
import logging
import importlib.metadata
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, List, Dict, Any, Optional

# Import and load dotenv if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv is optional - continue without it
    pass

import click
import mcp
import mcp.types as types
from mcp.server.fastmcp import FastMCP
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.fastmcp import Context

from mcp_server_qdrant.embeddings.factory import create_embedding_provider
from mcp_server_qdrant.qdrant import Entry, Metadata, QdrantConnector
from mcp_server_qdrant.settings import (
    EmbeddingProviderSettings,
    QdrantSettings,
    ToolSettings,
)
from mcp_server_qdrant.tools import (
    # Search tools
    natural_language_query,
    hybrid_search,
    multi_vector_search,
    
    # Collection management tools
    create_collection,
    migrate_collection,
    
    # Data processing tools
    batch_embed,
    chunk_and_process,
    
    # Advanced query tools
    semantic_router,
    decompose_query,
    
    # Analytics tools
    analyze_collection,
    benchmark_query,
    
    # Document processing tools
    index_document,
    process_pdf,
)

logger = logging.getLogger(__name__)


def get_package_version() -> str:
    """Get the package version using importlib.metadata."""
    try:
        return importlib.metadata.version("mcp-server-qdrant")
    except importlib.metadata.PackageNotFoundError:
        # Fall back to a default version if package is not installed
        return "0.0.0"


# Load the tool settings from the env variables, if they are set,
# or use the default values otherwise.
tool_settings = ToolSettings()

# Create a FastMCP instance to use for tool registration
fast_mcp = FastMCP("Qdrant MCP")


@fast_mcp.tool(name="qdrant-store", description=tool_settings.tool_store_description)
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


@fast_mcp.tool(name="qdrant-find", description=tool_settings.tool_find_description)
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


# Register all the custom tools

# Search tools
@fast_mcp.tool(name="nlq-search", description=tool_settings.nlq_search_description)
async def nlq_search(ctx: Context, **kwargs) -> Dict[str, Any]:
    return await natural_language_query(ctx, **kwargs)

@fast_mcp.tool(name="hybrid-search", description=tool_settings.hybrid_search_description)
async def hybrid_search_tool(ctx: Context, **kwargs) -> Dict[str, Any]:
    return await hybrid_search(ctx, **kwargs)

@fast_mcp.tool(name="multi-vector-search", description=tool_settings.multi_vector_search_description)
async def multi_vector_search_tool(ctx: Context, **kwargs) -> Dict[str, Any]:
    return await multi_vector_search(ctx, **kwargs)

# Collection management tools
@fast_mcp.tool(name="create-collection", description=tool_settings.create_collection_description)
async def create_collection_tool(ctx: Context, **kwargs) -> Dict[str, Any]:
    return await create_collection(ctx, **kwargs)

@fast_mcp.tool(name="migrate-collection", description=tool_settings.migrate_collection_description)
async def migrate_collection_tool(ctx: Context, **kwargs) -> Dict[str, Any]:
    return await migrate_collection(ctx, **kwargs)

# Data processing tools
@fast_mcp.tool(name="batch-embed", description=tool_settings.batch_embed_description)
async def batch_embed_tool(ctx: Context, **kwargs) -> Dict[str, Any]:
    return await batch_embed(ctx, **kwargs)

@fast_mcp.tool(name="chunk-and-process", description=tool_settings.chunk_and_process_description)
async def chunk_and_process_tool(ctx: Context, **kwargs) -> Dict[str, Any]:
    return await chunk_and_process(ctx, **kwargs)

# Advanced query tools
@fast_mcp.tool(name="semantic-router", description=tool_settings.semantic_router_description)
async def semantic_router_tool(ctx: Context, **kwargs) -> Dict[str, Any]:
    return await semantic_router(ctx, **kwargs)

@fast_mcp.tool(name="decompose-query", description=tool_settings.decompose_query_description)
async def decompose_query_tool(ctx: Context, **kwargs) -> Dict[str, Any]:
    return await decompose_query(ctx, **kwargs)

# Analytics tools
@fast_mcp.tool(name="analyze-collection", description=tool_settings.analyze_collection_description)
async def analyze_collection_tool(ctx: Context, **kwargs) -> Dict[str, Any]:
    return await analyze_collection(ctx, **kwargs)

@fast_mcp.tool(name="benchmark-query", description=tool_settings.benchmark_query_description)
async def benchmark_query_tool(ctx: Context, **kwargs) -> Dict[str, Any]:
    return await benchmark_query(ctx, **kwargs)

# Document processing tools
@fast_mcp.tool(name="index-document", description=tool_settings.index_document_description)
async def index_document_tool(ctx: Context, **kwargs) -> Dict[str, Any]:
    return await index_document(ctx, **kwargs)

@fast_mcp.tool(name="process-pdf", description=tool_settings.process_pdf_description)
async def process_pdf_tool(ctx: Context, **kwargs) -> Dict[str, Any]:
    return await process_pdf(ctx, **kwargs)


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

    # Note: The main entry point needs to be updated to use the new approach with tools
    import asyncio
    
    async def _run():
        # Setup logging
        logging.basicConfig(level=logging.DEBUG)
        logger.debug("Starting Qdrant MCP server...")
        
        # Load settings from environment
        embedding_settings = EmbeddingProviderSettings()
        
        # Create the embedding provider
        logger.debug(f"Creating embedding provider: {embedding_provider}")
        provider = create_embedding_provider(
            provider_type=embedding_provider or embedding_settings.provider_type,
            model_name=embedding_model or embedding_settings.model_name,
        )
        logger.debug("Embedding provider created successfully")
        
        # Create the Qdrant connector directly instead of using QdrantSettings
        logger.debug(f"Connecting to Qdrant: URL={qdrant_url}, Collection={collection_name}, Local path={qdrant_local_path}")
        qdrant_connector = QdrantConnector(
            qdrant_url,
            qdrant_api_key,
            collection_name,
            provider,
            qdrant_local_path,
        )
        logger.debug("Qdrant connector created successfully")
        
        @asynccontextmanager
        async def server_lifespan(mcp_server) -> AsyncIterator[Dict[str, Any]]:
            try:
                yield {"qdrant_connector": qdrant_connector}
            finally:
                # Cleanup if needed
                pass
        
        # Set the lifespan context manager
        logger.debug("Setting up lifespan context manager")
        fast_mcp.settings.lifespan = server_lifespan
        
        # We can't use fast_mcp.run() here because we're already in an asyncio context
        # Instead, let's run the stdio server directly
        logger.debug("Starting stdio server")
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            logger.debug("Stdio server started successfully, starting MCP server")
            await fast_mcp._mcp_server.run(
                read_stream,
                write_stream,
                fast_mcp._mcp_server.create_initialization_options(),
            )

    asyncio.run(_run())
