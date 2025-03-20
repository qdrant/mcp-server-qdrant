"""
Hybrid Search tool for Qdrant MCP server.
Combines vector search with keyword filtering for improved accuracy.
"""
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import Context

from mcp_server_qdrant.qdrant import QdrantConnector
from mcp_server_qdrant.tools.search.nlq import natural_language_query
from mcp_server_qdrant.tools.utils import build_keyword_filter, re_rank_results


async def hybrid_search(
    ctx: Context,
    query: str,
    collection: Optional[str] = None,
    limit: int = 10,
    filter: Optional[Dict[str, Any]] = None,
    text_field_name: str = "document",
) -> List[Dict[str, Any]]:
    """
    Perform hybrid vector and keyword search.
    
    This tool combines semantic vector search with keyword-based filtering
    to improve search accuracy, especially for queries that contain specific
    terms or phrases.
    
    Parameters:
    -----------
    ctx : Context
        The MCP request context
    query : str
        The search query
    collection : str, optional
        The name of the collection to search (overrides the default)
    limit : int, default=10
        Maximum number of results to return
    filter : Dict[str, Any], optional
        Additional Qdrant filters to apply
    text_field_name : str, default="document"
        The name of the field containing the document text
        
    Returns:
    --------
    List[Dict[str, Any]]
        Combined and re-ranked results from both search methods
    """
    await ctx.debug(f"Performing hybrid search: {query}")
    
    # Get the Qdrant connector from context
    qdrant_connector: QdrantConnector = ctx.request_context.lifespan_context[
        "qdrant_connector"
    ]
    
    # Use the specified collection or fall back to the default
    collection_name = collection or qdrant_connector._collection_name
    
    # Vector search - get more results than requested for re-ranking
    vector_limit = limit * 3
    vector_results = await natural_language_query(
        ctx=ctx,
        query=query,
        collection=collection_name,
        limit=vector_limit,
        filter=filter,
    )
    
    # Keyword search
    keyword_filter = build_keyword_filter(query, text_field_name)
    if filter:
        # Combine with any existing filters
        if "must" in filter:
            filter["must"].append(keyword_filter)
        else:
            filter["must"] = [keyword_filter]
    else:
        filter = {"must": [keyword_filter]}
    
    # Perform the keyword search
    client = qdrant_connector._client
    keyword_results_raw = await client.scroll(
        collection_name=collection_name,
        filter=filter,
        limit=vector_limit,
        with_payload=True,
    )
    
    # Format keyword results to match vector results format
    keyword_results = []
    for result in keyword_results_raw[0]:
        keyword_results.append({
            "score": 1.0,  # No score from scroll, using 1.0 as placeholder
            "document": result.payload.get("document", ""),
            "metadata": result.payload.get("metadata", {}),
            "id": str(result.id),
        })
    
    # Combine and re-rank
    combined_results = re_rank_results(vector_results, keyword_results, limit)
    
    return combined_results
