"""
Collection Analytics tool for Qdrant MCP server.
"""
from typing import Any, Dict, Optional

from mcp.server.fastmcp import Context

from mcp_server_qdrant.tools.utils import derive_payload_schema, calculate_vector_stats


async def analyze_collection(
    ctx: Context,
    collection: str,
    sample_size: int = 100,
) -> Dict[str, Any]:
    """
    Analyze a Qdrant collection to extract statistics and schema information.
    
    Parameters:
    -----------
    ctx : Context
        The MCP request context
    collection : str
        Name of the collection to analyze
    sample_size : int, default=100
        Number of points to sample for analysis
        
    Returns:
    --------
    Dict[str, Any]
        Collection statistics and schema information
    """
    await ctx.debug(f"Analyzing collection: {collection}, sample size: {sample_size}")
    
    # Get the Qdrant client from context
    qdrant_connector = ctx.request_context.lifespan_context["qdrant_connector"]
    client = qdrant_connector._client
    
    try:
        # Get collection info
        collection_info = await client.get_collection(collection_name=collection)
        
        # Sample points for analysis
        points_response = await client.scroll(
            collection_name=collection,
            limit=sample_size,
            with_vectors=True  # Include vectors for statistics
        )
        
        points = points_response.points
        await ctx.debug(f"Sampled {len(points)} points for analysis")
        
        # Extract payload schema
        payload_schema = derive_payload_schema(points)
        
        # Calculate vector statistics
        vector_stats = calculate_vector_stats(points)
        
        # Get collection counts
        if hasattr(collection_info, "vectors_count"):
            vectors_count = collection_info.vectors_count
        else:
            vectors_count = 0
            for shard in collection_info.shards:
                if hasattr(shard, "points_count"):
                    vectors_count += shard.points_count
        
        # Prepare result
        result = {
            "status": "success",
            "collection_name": collection,
            "count": vectors_count,
            "payload_schema": payload_schema,
            "vector_stats": vector_stats,
            "collection_info": {
                "vectors_config": collection_info.config.params.vectors,
                "shard_number": collection_info.config.params.shard_number,
                "replication_factor": collection_info.config.params.replication_factor,
                "write_consistency_factor": collection_info.config.params.write_consistency_factor,
                "on_disk_payload": collection_info.config.params.on_disk_payload,
            }
        }
        
        # Add index info if available
        if hasattr(collection_info.config.params, "hnsw_config"):
            result["collection_info"]["hnsw_config"] = {
                "m": collection_info.config.params.hnsw_config.m,
                "ef_construct": collection_info.config.params.hnsw_config.ef_construct,
                "full_scan_threshold": collection_info.config.params.hnsw_config.full_scan_threshold,
            }
        
        # Add optimizer info if available
        if hasattr(collection_info.config.params, "optimizers_config"):
            result["collection_info"]["optimizers_config"] = {
                "memmap_threshold": collection_info.config.params.optimizers_config.memmap_threshold,
                "indexing_threshold": collection_info.config.params.optimizers_config.indexing_threshold,
                "flush_interval_sec": collection_info.config.params.optimizers_config.flush_interval_sec,
            }
        
        return result
    except Exception as e:
        await ctx.debug(f"Error analyzing collection: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to analyze collection: {str(e)}"
        }
