"""
Collection Migration tool for Qdrant MCP server.
"""
import time
from typing import Any, Dict, Optional, List

from mcp.server.fastmcp import Context


async def migrate_collection(
    ctx: Context,
    source_collection: str,
    target_collection: str,
    batch_size: int = 100,
    transform_fn: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Migrate data between collections with optional transformations.
    
    Parameters:
    -----------
    ctx : Context
        The MCP request context
    source_collection : str
        Name of the source collection
    target_collection : str
        Name of the target collection
    batch_size : int, default=100
        Number of points to process in each batch
    transform_fn : str, optional
        Python code as string to transform each point
        
    Returns:
    --------
    Dict[str, Any]
        Status of the migration
    """
    await ctx.debug(f"Migrating from '{source_collection}' to '{target_collection}'")
    
    # Get the Qdrant connector and client from context
    qdrant_connector = ctx.request_context.lifespan_context["qdrant_connector"]
    client = qdrant_connector._client
    
    # Create a transformation function from string if provided
    transform_function = None
    if transform_fn:
        try:
            # Using exec is not ideal but works for user-provided transformation code
            # In a production system, consider safer alternatives
            transform_code = f"def transform_point(point):\n"
            for line in transform_fn.strip().split("\n"):
                transform_code += f"    {line}\n"
            
            transform_code += "transform_function = transform_point"
            
            # Create a local namespace to execute the code
            local_namespace = {}
            exec(transform_code, {}, local_namespace)
            
            transform_function = local_namespace["transform_function"]
            await ctx.debug("Successfully compiled transformation function")
        except Exception as e:
            await ctx.debug(f"Error compiling transformation function: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to compile transformation function: {str(e)}"
            }
    
    # If no custom function, use identity function
    if not transform_function:
        transform_function = lambda point: point
    
    # Verify that collections exist
    try:
        await client.get_collection(source_collection)
        await client.get_collection(target_collection)
    except Exception as e:
        await ctx.debug(f"Error verifying collections: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to verify collections: {str(e)}"
        }
    
    # Paginated migration
    try:
        offset = None
        total_migrated = 0
        start_time = time.time()
        
        while True:
            # Fetch batch from source
            points_response = await client.scroll(
                collection_name=source_collection,
                limit=batch_size,
                offset=offset
            )
            
            points = points_response.points
            
            if not points:
                break
            
            # Apply transformation
            transformed_points = []
            for point in points:
                try:
                    transformed_point = transform_function(point)
                    transformed_points.append(transformed_point)
                except Exception as e:
                    await ctx.debug(f"Error transforming point {point.id}: {str(e)}")
                    # Continue with other points
            
            # Upload to target
            if transformed_points:
                await client.upsert(
                    collection_name=target_collection,
                    points=transformed_points
                )
            
            # Update offset for pagination
            offset = points[-1].id
            total_migrated += len(transformed_points)
            
            await ctx.debug(f"Migrated {total_migrated} points so far")
        
        elapsed_time = time.time() - start_time
        
        return {
            "status": "success",
            "message": "Migration completed successfully",
            "points_migrated": total_migrated,
            "time_taken_seconds": round(elapsed_time, 2)
        }
    except Exception as e:
        await ctx.debug(f"Error during migration: {str(e)}")
        return {
            "status": "error",
            "message": f"Migration failed: {str(e)}"
        }
