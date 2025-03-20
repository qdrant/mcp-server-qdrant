"""
Semantic Router tool for Qdrant MCP server.
"""
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import Context

from mcp_server_qdrant.tools.utils import calculate_cosine_similarity


async def semantic_router(
    ctx: Context,
    query: str,
    routes: List[Dict[str, str]],
    threshold: Optional[float] = 0.6,
) -> Dict[str, Any]:
    """
    Route a query to the most semantically similar destination.
    
    Parameters:
    -----------
    ctx : Context
        The MCP request context
    query : str
        The query to route
    routes : List[Dict[str, str]]
        List of routes with 'name' and 'description' fields
    threshold : float, optional
        Minimum similarity threshold to consider a route match
        
    Returns:
    --------
    Dict[str, Any]
        Selected route and confidence
    """
    await ctx.debug(f"Routing query: {query}")
    
    # Validate routes format
    for i, route in enumerate(routes):
        if "name" not in route or "description" not in route:
            await ctx.debug(f"Invalid route format at index {i}: {route}")
            return {
                "status": "error",
                "message": "All routes must have 'name' and 'description' fields"
            }
    
    # Get the embedding provider from context
    qdrant_connector = ctx.request_context.lifespan_context["qdrant_connector"]
    embedding_provider = qdrant_connector._embedding_provider
    
    try:
        # Generate embedding for the query
        query_embedding = await embedding_provider.embed_query(query)
        
        # Generate embeddings for each route description
        route_embeddings = []
        for route in routes:
            route_embedding = await embedding_provider.embed_query(route["description"])
            route_embeddings.append(route_embedding)
        
        # Calculate cosine similarity between query and each route
        similarities = []
        for route_embedding in route_embeddings:
            similarity = calculate_cosine_similarity(query_embedding, route_embedding)
            similarities.append(similarity)
        
        # Find the best matching route
        if not similarities:
            return {
                "status": "error",
                "message": "No valid routes found"
            }
        
        max_index = similarities.index(max(similarities))
        max_similarity = similarities[max_index]
        
        # Apply threshold if specified
        if threshold is not None and max_similarity < threshold:
            return {
                "status": "success",
                "route": None,
                "confidence": max_similarity,
                "message": f"No route matched above threshold {threshold}"
            }
        
        # Return the best matching route
        return {
            "status": "success",
            "route": routes[max_index]["name"],
            "confidence": max_similarity,
            "description": routes[max_index]["description"]
        }
    except Exception as e:
        await ctx.debug(f"Error in semantic router: {str(e)}")
        return {
            "status": "error",
            "message": f"Routing failed: {str(e)}"
        }
