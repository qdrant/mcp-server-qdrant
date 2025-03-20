"""
Query Performance Benchmarking tool for Qdrant MCP server.
"""
import time
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import Context

from mcp_server_qdrant.tools.search.nlq import natural_language_query


async def benchmark_query(
    ctx: Context,
    query: str,
    collection: str,
    iterations: int = 5,
) -> Dict[str, Any]:
    """
    Benchmark query performance by running multiple iterations.
    
    Parameters:
    -----------
    ctx : Context
        The MCP request context
    query : str
        Query to benchmark
    collection : str
        Name of the collection to query
    iterations : int, default=5
        Number of iterations to run
        
    Returns:
    --------
    Dict[str, Any]
        Benchmark results with timing statistics
    """
    await ctx.debug(f"Benchmarking query: {query}, iterations: {iterations}")
    
    try:
        times = []
        results_count = []
        
        for i in range(iterations):
            # Run the query and measure time
            start_time = time.time()
            results = await natural_language_query(
                ctx=ctx,
                query=query,
                collection=collection
            )
            elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            times.append(elapsed_time)
            results_count.append(len(results))
            
            await ctx.debug(f"Iteration {i+1}: {elapsed_time:.2f} ms, {len(results)} results")
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        # Include deviation
        if len(times) > 1:
            variance = sum((t - avg_time) ** 2 for t in times) / len(times)
            std_dev = variance ** 0.5
        else:
            std_dev = 0
        
        return {
            "status": "success",
            "query": query,
            "collection": collection,
            "iterations": iterations,
            "performance": {
                "avg_time_ms": avg_time,
                "min_time_ms": min_time,
                "max_time_ms": max_time,
                "std_dev_ms": std_dev,
                "times_ms": times
            },
            "results": {
                "avg_count": sum(results_count) / len(results_count),
                "counts": results_count
            }
        }
    except Exception as e:
        await ctx.debug(f"Error benchmarking query: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to benchmark query: {str(e)}"
        }
