import asyncio
import importlib.metadata
import json
from typing import Optional, Dict, Any, List

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
        collection_name_description = (
            "Optional collection name to store the memory in. If not provided, the default collection will be used. "
            "Collection names must contain only alphanumeric characters, underscores, and hyphens (a-z, A-Z, 0-9, _, -), "
            "and be between 1 and 64 characters long. Examples: 'work', 'personal_notes', 'project-2023'."
        )
        
        # Check if all collections are protected from deletion
        all_collections_deletion_protected = "*" in qdrant_connector._protected_collections
        
        tools = [
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
                            "description": "The query to search for in the memories",
                        },
                        **({"collection_name": {
                            "type": "string",
                            "description": collection_name_description,
                        }} if qdrant_connector._multi_collection_mode else {}),
                    },
                    "required": ["query"],
                },
            ),
            # Always add the store memory tool since collections are now insert-only, not read-only
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
                        **({"replace_memory_ids": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "Optional list of memory IDs to replace with this new memory.",
                        }} if not all_collections_deletion_protected else {}),
                        **({"collection_name": {
                            "type": "string",
                            "description": collection_name_description
                        }} if qdrant_connector._multi_collection_mode else {}),
                    },
                    "required": ["information"],
                },
            )
        ]
        
        # Only add the delete memories tool if not all collections are protected from deletion
        if not all_collections_deletion_protected:
            tools.append(
                types.Tool(
                    name="qdrant-delete-memories",
                    description=(
                        "Delete specific memories from Qdrant. Use this tool when you need to: \n"
                        " - Remove outdated or incorrect information \n"
                        " - Update information by deleting old versions \n"
                        " - Clean up unnecessary memories"
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "memory_ids": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                },
                                "description": "List of memory IDs to delete. These IDs are returned by the find-memories tools.",
                            },
                        },
                        "required": ["memory_ids"],
                    },
                )
            )
        
        # Add multi-collection tools if enabled
        if qdrant_connector._multi_collection_mode:
            tools.extend([
                types.Tool(
                    name="qdrant-list-collections",
                    description=(
                        "List all available collections in Qdrant. Use this tool when you need to: \n"
                        " - See what collections are available \n"
                        " - Decide which collection to use for storing or retrieving memories"
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {},
                    },
                ),
                types.Tool(
                    name="qdrant-find-memories-across-collections",
                    description=(
                        "Look up memories across all collections in Qdrant. Use this tool when you need to: \n"
                        " - Find memories across different contexts \n"
                        " - Get a comprehensive view of all stored memories related to a query"
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The query to search for across all collections",
                            },
                        },
                        "required": ["query"],
                    },
                ),
            ])
        
        return tools

    @server.call_tool()
    async def handle_tool_call(
        name: str, arguments: dict | None
    ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        # Check if all collections are protected from deletion
        all_collections_deletion_protected = "*" in qdrant_connector._protected_collections
        
        valid_tools = ["qdrant-store-memory", "qdrant-find-memories"]
        
        # Only add the delete memories tool if not all collections are protected from deletion
        if not all_collections_deletion_protected:
            valid_tools.append("qdrant-delete-memories")
        
        # Add multi-collection tools if enabled
        if qdrant_connector._multi_collection_mode:
            valid_tools.extend(["qdrant-list-collections", "qdrant-find-memories-across-collections"])
            
        if name not in valid_tools:
            if name == "qdrant-delete-memories" and all_collections_deletion_protected:
                return [types.TextContent(type="text", text="Error: All collections are protected from deletion operations (insert-only).")]
            raise ValueError(f"Unknown tool: {name}")

        if name == "qdrant-store-memory":
            if not arguments or "information" not in arguments:
                raise ValueError("Missing required argument 'information'")
            
            information = arguments["information"]
            collection_name = arguments.get("collection_name") if qdrant_connector._multi_collection_mode else None
            replace_memory_ids = arguments.get("replace_memory_ids", [])
            
            try:
                memory_id = await qdrant_connector.store_memory(
                    information, 
                    collection_name,
                    replace_memory_ids
                )
                
                response = f"Remembered: {information}"
                if collection_name:
                    response += f" (in collection: {collection_name})"
                response += f"\nMemory ID: {memory_id}"
                    
                return [types.TextContent(type="text", text=response)]
            except ValueError as e:
                # Return a more user-friendly error message
                return [types.TextContent(type="text", text=f"Error: {str(e)}")]

        if name == "qdrant-find-memories":
            if not arguments or "query" not in arguments:
                raise ValueError("Missing required argument 'query'")
            
            query = arguments["query"]
            collection_name = arguments.get("collection_name") if qdrant_connector._multi_collection_mode else None
            
            try:
                memories = await qdrant_connector.find_memories(query, collection_name)
                
                content = [
                    types.TextContent(
                        type="text", 
                        text=f"Memories for the query '{query}'" + 
                             (f" in collection '{collection_name}'" if collection_name else "")
                    ),
                ]
                
                if not memories:
                    content.append(
                        types.TextContent(type="text", text="No memories found.")
                    )
                else:
                    # If all collections are protected, don't show readonly tags
                    all_collections_deletion_protected = "*" in qdrant_connector._protected_collections
                    
                    for memory in memories:
                        # Only show readonly tag if not all collections are protected
                        readonly_tag = " readonly" if memory.get("readonly", False) and not all_collections_deletion_protected else ""
                        content.append(
                            types.TextContent(
                                type="text", 
                                text=f"<memory id=\"{memory['id']}\"{readonly_tag}>{memory['content']}</memory>"
                            )
                        )
                    
                return content
            except ValueError as e:
                # Return a more user-friendly error message
                return [types.TextContent(type="text", text=f"Error: {str(e)}")]
            
        if name == "qdrant-delete-memories":
            if not arguments or "memory_ids" not in arguments:
                raise ValueError("Missing required argument 'memory_ids'")
            
            memory_ids = arguments["memory_ids"]
            
            try:
                deleted_count = await qdrant_connector.delete_memories(memory_ids)
                
                response = f"Successfully deleted {deleted_count} memories."
                if deleted_count < len(memory_ids):
                    skipped_count = len(memory_ids) - deleted_count
                    response += f" ({skipped_count} memories were not found or in protected collections.)"
                    
                return [types.TextContent(type="text", text=response)]
            except ValueError as e:
                # Return a more user-friendly error message
                return [types.TextContent(type="text", text=f"Error: {str(e)}")]
            
        if name == "qdrant-list-collections":
            collections = await qdrant_connector.list_collections()
            
            content = [
                types.TextContent(
                    type="text", 
                    text=f"Available collections:"
                ),
            ]
            
            if not collections:
                content.append(
                    types.TextContent(type="text", text="No collections found.")
                )
            else:
                for collection in collections:
                    collection_extra = ""
                    
                    if collection == qdrant_connector._collection_name:
                        collection_extra = " (default)"

                    if qdrant_connector._is_collection_deletion_protected(collection):
                        collection_extra += " (insert-only)"

                    if qdrant_connector._is_collection_readonly(collection):
                        collection_extra += " (readonly)"

                    content.append(
                        types.TextContent(type="text", text=f"- {collection}{collection_extra}")
                    )
                    
            return content
            
        if name == "qdrant-find-memories-across-collections":
            if not arguments or "query" not in arguments:
                raise ValueError("Missing required argument 'query'")
                
            query = arguments["query"]
            results = await qdrant_connector.find_memories_across_collections(query)
            
            content = [
                types.TextContent(
                    type="text", 
                    text=f"Memories across collections for the query '{query}':"
                ),
            ]
            
            if not results:
                content.append(
                    types.TextContent(type="text", text="No memories found in any collection.")
                )
            else:
                # If all collections are protected, don't show readonly tags
                all_collections_deletion_protected = "*" in qdrant_connector._protected_collections
                
                for collection, memories in results.items():
                    collection_extra = ""

                    if collection == qdrant_connector._collection_name:
                        collection_extra = " (default)"

                    if qdrant_connector._is_collection_deletion_protected(collection):
                        collection_extra += " (insert-only)"

                    if qdrant_connector._is_collection_readonly(collection):
                        collection_extra += " (readonly)"

                    content.append(
                        types.TextContent(type="text", text=f"\nCollection: {collection}{collection_extra}")
                    )
                    
                    for memory in memories:
                        # Only show readonly tag if not all collections are protected
                        readonly_tag = " readonly" if memory.get("readonly", False) and not all_collections_deletion_protected else ""
                        content.append(
                            types.TextContent(
                                type="text", 
                                text=f"<memory id=\"{memory['id']}\"{readonly_tag}>{memory['content']}</memory>"
                            )
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
    help="Collection name (used as the default collection in multi-collection mode)",
)
@click.option(
    "--fastembed-model-name",
    envvar="FASTEMBED_MODEL_NAME",
    required=False,
    help="FastEmbed model name"
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
@click.option(
    "--multi-collection-mode",
    envvar="MULTI_COLLECTION_MODE",
    is_flag=True,
    help="Enable multi-collection mode for AI agents",
)
@click.option(
    "--collection-prefix",
    envvar="COLLECTION_PREFIX",
    required=False,
    help="Prefix for all collections in multi-collection mode",
    default="agent_",
)
@click.option(
    "--protect-collections",
    envvar="PROTECT_COLLECTIONS",
    required=False,
    help="Comma-separated list of collection names that are protected from deletion operations (insert-only). In multi-collection mode, if specified without values, all collections will be insert-only.",
    is_flag=False,
    flag_value="",
)
def main(
    qdrant_url: Optional[str],
    qdrant_api_key: str,
    collection_name: Optional[str],
    fastembed_model_name: Optional[str],
    embedding_provider: str,
    embedding_model: str,
    qdrant_local_path: Optional[str],
    multi_collection_mode: bool,
    collection_prefix: Optional[str],
    protect_collections: Optional[str],
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
        
    # Parse protected collections
    protected_collections = set()
    if protect_collections is not None:
        # If protect_collections is specified but empty in multi-collection mode,
        # we'll protect all collections (determined dynamically)
        if protect_collections == "" and multi_collection_mode:
            # We'll set a flag to indicate that all collections should be protected
            # The actual collection names will be determined at runtime
            protected_collections = {"*"}  # Special marker for "all collections"
        else:
            # Otherwise, protect the specified collections
            protected_collections = set(name.strip() for name in protect_collections.split(",") if name.strip())
            
        # If not in multi-collection mode and protect_collections is specified,
        # assume the default collection should be protected
        if not multi_collection_mode and protected_collections:
            protected_collections = {collection_name}
            
    async def _run():
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            # Create the embedding provider
            provider = create_embedding_provider(
                provider_type=embedding_provider,
                model_name=embedding_model or fastembed_model_name,
            )

            # Create the Qdrant connector
            qdrant_connector = QdrantConnector(
                qdrant_url=qdrant_url,
                qdrant_api_key=qdrant_api_key,
                collection_name=collection_name,
                embedding_provider=provider,
                qdrant_local_path=qdrant_local_path,
                multi_collection_mode=multi_collection_mode,
                collection_prefix=collection_prefix,
                protected_collections=protected_collections,
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
