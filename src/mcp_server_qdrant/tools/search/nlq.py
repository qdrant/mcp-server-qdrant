"""
Natural Language Query tool for Qdrant MCP server.
"""
from typing import Any, Dict, List, Optional, Union

from mcp.server.fastmcp import Context

from mcp_server_qdrant.qdrant import Entry, QdrantConnector


async def natural_language_query(
    ctx: Context,
    query: str,
    collection: Optional[str] = None,
    limit: int = 10,
    filter: Optional[Dict[str, Any]] = None,
    with_payload: Union[bool, List[str]] = True,
) -> List[Dict[str, Any]]:
    """
    Search Qdrant using natural language.
    
    This tool performs a semantic search using the provided query string.
    The query is converted to a vector embedding and compared against 
    documents in the collection.
    
    Parameters:
    -----------
    ctx : Context
        The MCP request context
    query : str
        The natural language query
    collection : str, optional
        The name of the collection to search (overrides the default)
    limit : int, default=10
        Maximum number of results to return
    filter : Dict[str, Any], optional
        Additional Qdrant filters to apply
    with_payload : Union[bool, List[str]], default=True
        Whether to include payload fields (True for all, list for specific fields)
        
    Returns:
    --------
    List[Dict[str, Any]]
        Matched documents with their scores and payloads
    """
    await ctx.debug(f"Performing natural language query: {query}")
    
    # Get the Qdrant connector from context
    qdrant_connector: QdrantConnector = ctx.request_context.lifespan_context[
        "qdrant_connector"
    ]
    
    # Use the specified collection or fall back to the default
    collection_name = collection or qdrant_connector._collection_name
    
    # Embed the query
    embedding_provider = qdrant_connector._embedding_provider
    query_vector = await embedding_provider.embed_query(query)
    vector_name = embedding_provider.get_vector_name()
    
    # Search in Qdrant
    search_results = await qdrant_connector._client.search(
        collection_name=collection_name,
        query_vector={vector_name: query_vector},
        limit=limit,
        filter=filter,
        with_payload=with_payload,
    )
    
    # Format results
    formatted_results = []
    for result in search_results:
        formatted_results.append({
            "score": result.score,
            "document": result.payload.get("document", ""),
            "metadata": result.payload.get("metadata", {}),
            "id": str(result.id),
        })
    
    return formatted_results
