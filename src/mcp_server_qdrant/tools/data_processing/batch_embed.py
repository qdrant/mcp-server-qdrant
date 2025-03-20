"""
Batch Embedding tool for Qdrant MCP server.
"""
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import Context


async def batch_embed(
    ctx: Context,
    texts: List[str],
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate embeddings for multiple texts in batch.
    
    Parameters:
    -----------
    ctx : Context
        The MCP request context
    texts : List[str]
        List of texts to embed
    model : str, optional
        Model name to use for embedding (overrides default)
        
    Returns:
    --------
    Dict[str, Any]
        Embeddings for each text
    """
    await ctx.debug(f"Generating embeddings for {len(texts)} texts")
    
    # Get the embedding provider from context
    qdrant_connector = ctx.request_context.lifespan_context["qdrant_connector"]
    embedding_provider = qdrant_connector._embedding_provider
    
    # If a specific model is requested, validate it's available
    if model and not embedding_provider.supports_model(model):
        await ctx.debug(f"Model {model} not supported, using default")
        model = None  # Fallback to default
    
    try:
        # Generate embeddings for all texts
        embeddings = []
        for text in texts:
            if model:
                embedding = await embedding_provider.embed_query(text, model=model)
            else:
                embedding = await embedding_provider.embed_query(text)
            embeddings.append(embedding)
        
        # Get the vector name (varies by embedding provider)
        vector_name = embedding_provider.get_vector_name()
        
        return {
            "status": "success",
            "embeddings": embeddings,
            "count": len(embeddings),
            "vector_name": vector_name,
            "dimensions": len(embeddings[0]) if embeddings else 0
        }
    except Exception as e:
        await ctx.debug(f"Error generating embeddings: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to generate embeddings: {str(e)}"
        }
