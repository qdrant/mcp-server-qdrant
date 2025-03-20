"""
Chunking and Processing tool for Qdrant MCP server.
"""
import time
import uuid
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import Context

from mcp_server_qdrant.tools.utils import text_to_chunks


async def chunk_and_process(
    ctx: Context,
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    collection: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Split text into chunks, generate embeddings, and optionally store in Qdrant.
    
    Parameters:
    -----------
    ctx : Context
        The MCP request context
    text : str
        Text to process and chunk
    chunk_size : int, default=1000
        Maximum size of each chunk
    chunk_overlap : int, default=200
        Overlap between consecutive chunks
    collection : str, optional
        Collection to store chunks in (if provided)
    metadata : Dict[str, Any], optional
        Additional metadata to include with each chunk
        
    Returns:
    --------
    Dict[str, Any]
        Processed chunks with embeddings
    """
    await ctx.debug(f"Processing text with chunk size {chunk_size}, overlap {chunk_overlap}")
    
    # Get dependencies from context
    qdrant_connector = ctx.request_context.lifespan_context["qdrant_connector"]
    embedding_provider = qdrant_connector._embedding_provider
    client = qdrant_connector._client
    
    # Create metadata dict if not provided
    if metadata is None:
        metadata = {}
    
    try:
        # Split text into chunks
        chunks = text_to_chunks(text, chunk_size, chunk_overlap)
        await ctx.debug(f"Split text into {len(chunks)} chunks")
        
        # Process each chunk
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            # Generate embedding
            embedding = await embedding_provider.embed_query(chunk)
            vector_name = embedding_provider.get_vector_name()
            
            # Create point object
            point_id = str(uuid.uuid4())
            chunk_metadata = {
                "chunk_index": i,
                "total_chunks": len(chunks),
                "created_at": time.time(),
                **metadata  # Add user-provided metadata
            }
            
            point = {
                "id": point_id,
                "vector": {vector_name: embedding} if isinstance(embedding, list) else embedding,
                "payload": {
                    "text": chunk,
                    **chunk_metadata
                }
            }
            
            processed_chunks.append(point)
        
        # Upload to collection if specified
        if collection and processed_chunks:
            try:
                # Check if collection exists
                await client.get_collection(collection)
                
                # Upload points
                await client.upsert(
                    collection_name=collection,
                    points=processed_chunks
                )
                await ctx.debug(f"Uploaded {len(processed_chunks)} chunks to collection '{collection}'")
            except Exception as e:
                await ctx.debug(f"Error uploading to collection: {str(e)}")
                return {
                    "status": "error",
                    "message": f"Failed to upload to collection: {str(e)}",
                    "chunks": processed_chunks
                }
        
        # Return processed chunks
        return {
            "status": "success",
            "chunks": processed_chunks,
            "count": len(processed_chunks),
            "metadata": metadata
        }
    except Exception as e:
        await ctx.debug(f"Error processing chunks: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to process text: {str(e)}"
        }
