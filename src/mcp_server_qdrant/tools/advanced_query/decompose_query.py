"""
Query Decomposition tool for Qdrant MCP server.
"""
import os
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import Context

from mcp_server_qdrant.tools.utils import simple_decompose, llm_decompose


async def decompose_query(
    ctx: Context,
    query: str,
    use_llm: bool = False,
) -> Dict[str, Any]:
    """
    Decompose a complex query into multiple simpler subqueries.
    
    Parameters:
    -----------
    ctx : Context
        The MCP request context
    query : str
        The complex query to decompose
    use_llm : bool, default=False
        Whether to use an LLM for decomposition (requires API key)
        
    Returns:
    --------
    Dict[str, Any]
        Decomposed subqueries
    """
    await ctx.debug(f"Decomposing query: {query}, use_llm: {use_llm}")
    
    try:
        if use_llm:
            # Use LLM for decomposition (requires API key)
            llm_api_key = os.environ.get("LLM_API_KEY")
            if not llm_api_key:
                await ctx.debug("LLM API key not found, falling back to rule-based decomposition")
                subqueries = await simple_decompose(query)
            else:
                await ctx.debug("Using LLM for query decomposition")
                subqueries = await llm_decompose(query, llm_api_key)
        else:
            # Use rule-based decomposition
            subqueries = await simple_decompose(query)
        
        await ctx.debug(f"Decomposed into {len(subqueries)} subqueries")
        
        return {
            "status": "success",
            "subqueries": subqueries,
            "count": len(subqueries),
            "original_query": query
        }
    except Exception as e:
        await ctx.debug(f"Error decomposing query: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to decompose query: {str(e)}"
        }
