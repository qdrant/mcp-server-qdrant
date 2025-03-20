"""
Collection Creation tool for Qdrant MCP server.
"""
from typing import Any, Dict, Optional, Union, Literal

from mcp.server.fastmcp import Context
from qdrant_client.http.models import Distance, VectorParams, HnswConfigDiff, OptimizersConfigDiff


async def create_collection(
    ctx: Context,
    name: str,
    vector_size: int,
    distance: Optional[Literal["Cosine", "Euclid", "Dot"]] = "Cosine",
    hnsw_config: Optional[Dict[str, Any]] = None,
    optimizers_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a new Qdrant collection with specified parameters.
    
    Parameters:
    -----------
    ctx : Context
        The MCP request context
    name : str
        Name for the new collection
    vector_size : int
        Size of the vector embeddings
    distance : str, optional
        Distance metric to use: "Cosine", "Euclid", or "Dot"
    hnsw_config : Dict[str, Any], optional
        Configuration for HNSW index
    optimizers_config : Dict[str, Any], optional
        Configuration for optimizers
        
    Returns:
    --------
    Dict[str, Any]
        Status of the collection creation
    """
    await ctx.debug(f"Creating collection: {name}")
    
    # Get the Qdrant connector from context
    qdrant_connector = ctx.request_context.lifespan_context["qdrant_connector"]
    client = qdrant_connector._client
    
    # Convert string distance to Qdrant enum
    distance_map = {
        "Cosine": Distance.COSINE,
        "Euclid": Distance.EUCLID,
        "Dot": Distance.DOT
    }
    distance_enum = distance_map.get(distance, Distance.COSINE)
    
    # Create HNSW config if provided
    hnsw_config_obj = None
    if hnsw_config:
        hnsw_config_obj = HnswConfigDiff(
            m=hnsw_config.get("m"),
            ef_construct=hnsw_config.get("ef_construct"),
            full_scan_threshold=hnsw_config.get("full_scan_threshold")
        )
    
    # Create optimizers config if provided
    optimizers_config_obj = None
    if optimizers_config:
        optimizers_config_obj = OptimizersConfigDiff(
            memmap_threshold=optimizers_config.get("memmap_threshold"),
            indexing_threshold=optimizers_config.get("indexing_threshold"),
            flush_interval_sec=optimizers_config.get("flush_interval_sec")
        )
    
    # Create the collection
    try:
        await client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=distance_enum
            ),
            hnsw_config=hnsw_config_obj,
            optimizers_config=optimizers_config_obj
        )
        
        return {
            "status": "success",
            "message": f"Collection '{name}' created successfully",
            "collection_name": name,
            "vector_size": vector_size,
            "distance": distance
        }
    except Exception as e:
        await ctx.debug(f"Error creating collection: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to create collection: {str(e)}"
        }
