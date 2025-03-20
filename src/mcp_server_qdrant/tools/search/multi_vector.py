"""
Multi-Vector Search tool for Qdrant MCP server.
Enables searching with multiple weighted query vectors.
"""
from typing import Any, Dict, List, Optional, Union

from mcp.server.fastmcp import Context

from mcp_server_qdrant.qdrant import QdrantConnector
from mcp_server_qdrant.tools.utils import combine_vectors


async def multi_vector_search(
    ctx: Context,
    queries: List[str],
    collection: Optional[str] = None,
    weights: Optional[List[float]] = None,
    limit: int = 10,
    filter: Optional[Dict[str, Any]] = None,
    with_payload: Union[bool, List[str]] = True,
) -> List[Dict[str, Any]]:
    """
    Search using multiple weighted query vectors.
    
    This tool allows combining multiple queries with different weights
    to find documents that match a complex combination of criteria.
    
    Parameters:
    -----------
    ctx : Context
        The MCP request context
    queries : List[str]
        List of query strings to embed and combine
    collection : str, optional
        The name of the collection to search (overrides the default)
    weights : List[float], optional
        Weights for each query (must match length of queries)
        If not provided, equal weights will be used
    limit : int, default=10
        Maximum number of results to return
    filter : Dict[str, Any], optional
        Additional Qdrant filters to apply
    with_payload : Union[bool, List[str]], default=True
        Whether to include payload fields
        
    Returns:
    --------
    List[Dict[str, Any]]
        Matched documents with their scores and payloads
    """
    await ctx.debug(f"Performing multi-vector search with {len(queries)} queries")
    
    # Get the Qdrant connector from context
    qdrant_connector: QdrantConnector = ctx.request_context.lifespan_context[
        "qdrant_connector"
    ]
    
    # Use the specified collection or fall back to the default
    collection_name = collection or qdrant_connector._collection_name
    
    # Generate embeddings for all queries
    embedding_provider = qdrant_connector._embedding_provider
    embeddings = await embedding_provider.embed_documents(queries)
    
    # Normalize weights if provided, or use equal weights
    if weights is None:
        weights = [1.0 / len(queries)] * len(queries)
    elif len(weights) != len(queries):
        raise ValueError("Number of weights must match number of queries")
    
    # Combine the vectors with weights
    combined_vector = combine_vectors(embeddings, weights)
    vector_name = embedding_provider.get_vector_name()
    
    # Search with the combined vector
    client = qdrant_connector._client
    search_results = await client.search(
        collection_name=collection_name,
        query_vector={vector_name: combined_vector},
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
