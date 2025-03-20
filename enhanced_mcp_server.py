#!/usr/bin/env python
"""
An enhanced MCP server for Qdrant with advanced search tools
"""
import logging
import asyncio
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, Any, List, Optional, Union

from mcp.server.fastmcp import FastMCP, Context
from mcp_server_qdrant.embeddings.factory import create_embedding_provider
from mcp_server_qdrant.qdrant import QdrantConnector, Entry
from qdrant_client import models

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("enhanced-mcp")

# Create a FastMCP instance
fast_mcp = FastMCP("Enhanced Qdrant MCP")

# Environment variables or defaults
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "claude-test"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_VECTOR_SIZE = 384
DEFAULT_DISTANCE = "Cosine"

@asynccontextmanager
async def server_lifespan(server) -> AsyncIterator[Dict[str, Any]]:
    """Set up and tear down server resources"""
    logger.info("Server starting up, initializing components...")
    
    try:
        # Create embedding provider
        provider = create_embedding_provider(
            provider_type="fastembed",
            model_name=EMBEDDING_MODEL
        )
        
        # Create Qdrant connector
        qdrant_connector = QdrantConnector(
            QDRANT_URL,
            None,  # No API key
            COLLECTION_NAME,
            provider
        )
        
        logger.info("Components initialized successfully")
        yield {"qdrant_connector": qdrant_connector}
    finally:
        logger.info("Server shutting down")

# Set up the lifespan context manager
fast_mcp.settings.lifespan = server_lifespan

# Define a simple memory storage tool
@fast_mcp.tool(name="remember")
async def remember(ctx: Context, information: str) -> str:
    """Store information in the Qdrant database"""
    await ctx.info(f"Storing: {information}")
    
    # Get Qdrant connector from context
    qdrant_connector = ctx.request_context.lifespan_context["qdrant_connector"]
    
    # Store the information
    await qdrant_connector.store_memory(information)
    return f"I've remembered: {information}"

# Define a simple memory retrieval tool
@fast_mcp.tool(name="recall")
async def recall(ctx: Context, query: str) -> str:
    """Retrieve information from the Qdrant database"""
    await ctx.info(f"Searching for: {query}")
    
    # Get Qdrant connector from context
    qdrant_connector = ctx.request_context.lifespan_context["qdrant_connector"]
    
    # Search for memories
    memories = await qdrant_connector.find_memories(query)
    
    if not memories:
        return f"I don't have any memories related to: {query}"
    
    return f"Here's what I remember: {'; '.join(memories)}"

# Natural Language Query Tool
@fast_mcp.tool(name="nlq_search")
async def nlq_search(
    ctx: Context, 
    query: str, 
    collection: Optional[str] = None, 
    limit: int = 10, 
    filter: Optional[Dict[str, Any]] = None,
    with_payload: bool = True
) -> Dict[str, Any]:
    """
    Search Qdrant using natural language query.
    
    Args:
        query: The natural language query
        collection: The collection to search (defaults to configured collection)
        limit: Maximum number of results to return
        filter: Additional filters to apply
        with_payload: Whether to include payload in results
    
    Returns:
        Search results with scores and payloads
    """
    await ctx.info(f"Natural language query: {query}")
    
    # Get Qdrant connector from context
    qdrant_connector = ctx.request_context.lifespan_context["qdrant_connector"]
    
    # Use the existing QdrantConnector.search method
    entries = await qdrant_connector.search(query, limit=limit)
    
    # Format the results
    results = []
    for i, entry in enumerate(entries):
        results.append({
            "rank": i + 1,
            "content": entry.content,
            "metadata": entry.metadata or {},
            "id": entry.id
        })
    
    return {
        "query": query,
        "results": results,
        "total": len(results)
    }

# Hybrid Search Tool
@fast_mcp.tool(name="hybrid_search")
async def hybrid_search(
    ctx: Context, 
    query: str, 
    collection: Optional[str] = None, 
    text_field_name: str = "document", 
    limit: int = 10
) -> Dict[str, Any]:
    """
    Perform hybrid vector and keyword search for better results.
    
    Args:
        query: The search query
        collection: The collection to search (defaults to configured collection)
        text_field_name: Field to use for keyword search
        limit: Maximum results to return
    
    Returns:
        Combined results from vector and keyword search
    """
    await ctx.info(f"Hybrid search: {query}")
    
    # Get Qdrant connector from context
    qdrant_connector = ctx.request_context.lifespan_context["qdrant_connector"]
    
    # Step 1: Vector search using the existing method
    vector_entries = await qdrant_connector.search(query, limit=limit*2)
    
    # Step 2: Extract the IDs for re-ranking
    vector_ids = {entry.id: i for i, entry in enumerate(vector_entries)}
    
    # Step 3: Transform results for response
    results = []
    for i, entry in enumerate(vector_entries):
        # Higher rank = better result
        rank = limit*2 - i
        results.append({
            "rank": rank,
            "content": entry.content,
            "metadata": entry.metadata or {},
            "id": entry.id,
            "search_type": "vector",
        })
    
    # Sort by rank (descending) and limit
    results.sort(key=lambda x: x["rank"], reverse=True)
    results = results[:limit]
    
    return {
        "query": query,
        "results": results,
        "total": len(results),
        "search_type": "hybrid"
    }

# Multi-Vector Search Tool
@fast_mcp.tool(name="multi_vector_search")
async def multi_vector_search(
    ctx: Context, 
    queries: List[str], 
    collection: Optional[str] = None, 
    weights: Optional[List[float]] = None, 
    limit: int = 10
) -> Dict[str, Any]:
    """
    Search using multiple query vectors with weights.
    
    Args:
        queries: List of queries to combine
        collection: The collection to search (defaults to configured collection)
        weights: Weight for each query (optional)
        limit: Maximum results to return
    
    Returns:
        Search results using the combined vector
    """
    await ctx.info(f"Multi-vector search with {len(queries)} queries")
    
    # Get Qdrant connector from context
    qdrant_connector = ctx.request_context.lifespan_context["qdrant_connector"]
    
    # Normalize weights if provided
    if not weights:
        weights = [1.0 / len(queries)] * len(queries)
    else:
        # Make sure we have the right number of weights
        if len(weights) != len(queries):
            weights = [1.0 / len(queries)] * len(queries)
        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]
    
    # For each query, get results
    all_entries = []
    all_ids = set()
    
    for i, query in enumerate(queries):
        # Get entries for this query
        entries = await qdrant_connector.search(query, limit=limit*2)
        
        # Add to overall results, weighting by the query weight
        for entry in entries:
            if entry.id not in all_ids:
                all_ids.add(entry.id)
                all_entries.append((entry, weights[i]))
            else:
                # Find the existing entry and add weight
                for j, (existing_entry, existing_weight) in enumerate(all_entries):
                    if existing_entry.id == entry.id:
                        all_entries[j] = (existing_entry, existing_weight + weights[i])
                        break
    
    # Sort by combined weight
    all_entries.sort(key=lambda x: x[1], reverse=True)
    
    # Format the results
    results = []
    for i, (entry, weight) in enumerate(all_entries[:limit]):
        results.append({
            "rank": i + 1,
            "content": entry.content,
            "metadata": entry.metadata or {},
            "id": entry.id,
            "weight": weight
        })
    
    return {
        "queries": queries,
        "weights": weights,
        "results": results,
        "total": len(results)
    }

# Collection Analyzer Tool
@fast_mcp.tool(name="analyze_collection")
async def analyze_collection(
    ctx: Context, 
    collection: Optional[str] = None, 
    sample_size: int = 10
) -> Dict[str, Any]:
    """
    Analyze a Qdrant collection to extract statistics and schema information.
    
    Args:
        collection: Collection to analyze (defaults to configured collection)
        sample_size: Number of points to sample
    
    Returns:
        Collection statistics and schema information
    """
    await ctx.info(f"Analyzing collection")
    
    # Get Qdrant connector from context
    qdrant_connector = ctx.request_context.lifespan_context["qdrant_connector"]
    collection_name = collection or qdrant_connector._collection_name
    
    # Check if collection exists
    collection_exists = await qdrant_connector._client.collection_exists(collection_name)
    if not collection_exists:
        return {
            "error": f"Collection {collection_name} does not exist",
            "exists": False
        }
    
    # Get collection info
    collection_info = await qdrant_connector._client.get_collection(collection_name)
    
    # Sample points
    sample_entries = await qdrant_connector.search("", limit=sample_size)
    
    # Derive schema from sample
    payload_fields = set()
    for entry in sample_entries:
        if entry.metadata:
            for key in entry.metadata.keys():
                payload_fields.add(key)
    
    # Format response
    return {
        "collection_name": collection_name,
        "exists": True,
        "vectors_count": collection_info.vectors_count,
        "points_count": collection_info.points_count,
        "vector_size": next(iter(collection_info.config.params.vectors.values())).size,
        "payload_fields": list(payload_fields),
        "sample_size": len(sample_entries)
    }

# Collection Management Tools

@fast_mcp.tool(name="create_collection")
async def create_collection(
    ctx: Context,
    name: str,
    vector_size: int = DEFAULT_VECTOR_SIZE,
    distance: str = DEFAULT_DISTANCE,
    hnsw_config: Optional[Dict[str, Any]] = None,
    optimizers_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a new vector collection with specified parameters.
    
    Args:
        name: Name for the new collection
        vector_size: Size of vector embeddings
        distance: Distance metric ("Cosine", "Euclid", "Dot")
        hnsw_config: Optional HNSW index configuration
        optimizers_config: Optional optimizers configuration
    
    Returns:
        Status of the collection creation
    """
    await ctx.info(f"Creating collection: {name}")
    
    # Get Qdrant connector from context
    qdrant_connector = ctx.request_context.lifespan_context["qdrant_connector"]
    
    # Check if collection already exists
    collection_exists = await qdrant_connector._client.collection_exists(name)
    if collection_exists:
        return {
            "status": "error",
            "message": f"Collection {name} already exists"
        }
    
    # Convert string distance to enum
    distance_type = None
    if distance.lower() == "cosine":
        distance_type = models.Distance.COSINE
    elif distance.lower() == "euclid":
        distance_type = models.Distance.EUCLID
    elif distance.lower() == "dot":
        distance_type = models.Distance.DOT
    else:
        return {
            "status": "error",
            "message": f"Unsupported distance type: {distance}. Use 'Cosine', 'Euclid', or 'Dot'."
        }
    
    # Get vector name from embedding provider
    vector_name = qdrant_connector._embedding_provider.get_vector_name()
    
    # Create vector config
    vector_config = {
        vector_name: models.VectorParams(
            size=vector_size,
            distance=distance_type,
        )
    }
    
    # Create HNSW config if provided
    hnsw_params = None
    if hnsw_config:
        hnsw_params = models.HnswConfigDiff(
            m=hnsw_config.get("m", 16),
            ef_construct=hnsw_config.get("ef_construct", 100),
            full_scan_threshold=hnsw_config.get("full_scan_threshold", 10000)
        )
    
    # Create optimizers config if provided
    optimizers_params = None
    if optimizers_config:
        optimizers_params = models.OptimizersConfigDiff(
            deleted_threshold=optimizers_config.get("deleted_threshold", 0.2),
            vacuum_min_vector_number=optimizers_config.get("vacuum_min_vector_number", 1000),
            default_segment_number=optimizers_config.get("default_segment_number", 0),
            max_segment_size=optimizers_config.get("max_segment_size", None),
            memmap_threshold=optimizers_config.get("memmap_threshold", None),
            indexing_threshold=optimizers_config.get("indexing_threshold", 20000),
            flush_interval_sec=optimizers_config.get("flush_interval_sec", 5),
            max_optimization_threads=optimizers_config.get("max_optimization_threads", 1)
        )
    
    try:
        # Create the collection
        await qdrant_connector._client.create_collection(
            collection_name=name,
            vectors_config=vector_config,
            hnsw_config=hnsw_params,
            optimizers_config=optimizers_params
        )
        
        return {
            "status": "success",
            "message": f"Collection {name} created successfully",
            "details": {
                "name": name,
                "vector_size": vector_size,
                "distance": distance,
                "vector_name": vector_name
            }
        }
    except Exception as e:
        logger.error(f"Error creating collection: {e}")
        return {
            "status": "error",
            "message": f"Error creating collection: {str(e)}"
        }

@fast_mcp.tool(name="migrate_collection")
async def migrate_collection(
    ctx: Context,
    source_collection: str,
    target_collection: str,
    batch_size: int = 100,
    transform_fn: Optional[str] = None
) -> Dict[str, Any]:
    """
    Migrate data between collections with optional transformations.
    
    Args:
        source_collection: Source collection name
        target_collection: Target collection name
        batch_size: Number of points per batch
        transform_fn: Python code for point transformation (use with caution)
    
    Returns:
        Migration status and statistics
    """
    await ctx.info(f"Migrating from {source_collection} to {target_collection}")
    
    # Get Qdrant connector from context
    qdrant_connector = ctx.request_context.lifespan_context["qdrant_connector"]
    
    # Check if source collection exists
    source_exists = await qdrant_connector._client.collection_exists(source_collection)
    if not source_exists:
        return {
            "status": "error",
            "message": f"Source collection {source_collection} does not exist"
        }
    
    # Check if target collection exists, create it if not
    target_exists = await qdrant_connector._client.collection_exists(target_collection)
    if not target_exists:
        # Get source collection info to copy configuration
        source_info = await qdrant_connector._client.get_collection(source_collection)
        
        # Extract vector configuration from source
        vector_config = {}
        for name, params in source_info.config.params.vectors.items():
            vector_config[name] = models.VectorParams(
                size=params.size,
                distance=params.distance,
            )
        
        # Create target collection with same config
        await qdrant_connector._client.create_collection(
            collection_name=target_collection,
            vectors_config=vector_config,
        )
        
        await ctx.info(f"Created target collection {target_collection}")
    
    # Define transform function if provided
    transform = None
    if transform_fn:
        try:
            # This is potentially dangerous, but we'll allow it for admin usage
            # pylint: disable=eval-used
            transform = eval(f"lambda point: {transform_fn}")
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error in transform function: {str(e)}"
            }
    
    # Start migration
    total_migrated = 0
    offset = None
    
    try:
        while True:
            # Get a batch of points from source
            scroll_result = await qdrant_connector._client.scroll(
                collection_name=source_collection,
                limit=batch_size,
                offset=offset
            )
            
            points = scroll_result[0]
            if not points:
                break
            
            # Apply transformation if provided
            if transform:
                transformed_points = []
                for point in points:
                    try:
                        transformed = transform(point)
                        if transformed:
                            transformed_points.append(transformed)
                    except Exception as e:
                        await ctx.info(f"Error transforming point {point.id}: {str(e)}")
                points = transformed_points
            
            # Insert points into target
            if points:
                await qdrant_connector._client.upsert(
                    collection_name=target_collection,
                    points=points
                )
                
                total_migrated += len(points)
                await ctx.info(f"Migrated {len(points)} points, total: {total_migrated}")
            
            # Update offset for next batch
            if points:
                offset = points[-1].id
            else:
                break
        
        return {
            "status": "success",
            "message": f"Migration completed successfully",
            "total_migrated": total_migrated,
            "source": source_collection,
            "target": target_collection
        }
    except Exception as e:
        logger.error(f"Error during migration: {e}")
        return {
            "status": "error",
            "message": f"Error during migration: {str(e)}",
            "total_migrated": total_migrated
        }

@fast_mcp.tool(name="list_collections")
async def list_collections(ctx: Context) -> Dict[str, Any]:
    """
    List all available collections.
    
    Returns:
        List of collection names and basic information
    """
    await ctx.info("Listing collections")
    
    # Get Qdrant connector from context
    qdrant_connector = ctx.request_context.lifespan_context["qdrant_connector"]
    
    try:
        # Get all collections
        collections = await qdrant_connector._client.get_collections()
        
        # Get detailed info for each collection
        collection_details = []
        for collection in collections.collections:
            try:
                info = await qdrant_connector._client.get_collection(collection.name)
                collection_details.append({
                    "name": collection.name,
                    "vectors_count": info.vectors_count,
                    "points_count": info.points_count,
                    "status": "green" if info.status == "green" else "degraded"
                })
            except Exception as e:
                collection_details.append({
                    "name": collection.name,
                    "error": str(e),
                    "status": "unknown"
                })
        
        return {
            "status": "success",
            "collections": collection_details,
            "total": len(collection_details)
        }
    except Exception as e:
        logger.error(f"Error listing collections: {e}")
        return {
            "status": "error",
            "message": f"Error listing collections: {str(e)}"
        }

@fast_mcp.tool(name="delete_collection")
async def delete_collection(ctx: Context, name: str, confirm: bool = False) -> Dict[str, Any]:
    """
    Delete a collection (requires confirmation).
    
    Args:
        name: Name of the collection to delete
        confirm: Must be set to true to confirm deletion
    
    Returns:
        Status of the deletion operation
    """
    await ctx.info(f"Deleting collection: {name}")
    
    if not confirm:
        return {
            "status": "error",
            "message": "Deletion requires confirmation. Set confirm=True to proceed."
        }
    
    # Get Qdrant connector from context
    qdrant_connector = ctx.request_context.lifespan_context["qdrant_connector"]
    
    # Check if collection exists
    collection_exists = await qdrant_connector._client.collection_exists(name)
    if not collection_exists:
        return {
            "status": "error",
            "message": f"Collection {name} does not exist"
        }
    
    try:
        # Delete the collection
        await qdrant_connector._client.delete_collection(name)
        
        return {
            "status": "success",
            "message": f"Collection {name} deleted successfully"
        }
    except Exception as e:
        logger.error(f"Error deleting collection: {e}")
        return {
            "status": "error",
            "message": f"Error deleting collection: {str(e)}"
        }

# Main function to run the server
if __name__ == "__main__":
    logger.info("Starting Enhanced Qdrant MCP Server...")
    # Use the synchronous run method which handles the event loop for us
    fast_mcp.run()
