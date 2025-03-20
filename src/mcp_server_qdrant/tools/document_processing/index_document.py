"""
Document Indexing tool for Qdrant MCP server.
"""
from datetime import datetime
from typing import Any, Dict, Optional

from mcp.server.fastmcp import Context

from mcp_server_qdrant.tools.utils import fetch_document, extract_text
from mcp_server_qdrant.tools.data_processing.chunk_and_process import chunk_and_process


async def index_document(
    ctx: Context,
    document_url: str,
    collection: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Fetch, process, and index a document from a URL.
    
    Parameters:
    -----------
    ctx : Context
        The MCP request context
    document_url : str
        URL of the document to fetch and index
    collection : str
        Collection to store the document chunks in
    chunk_size : int, default=1000
        Maximum size of each chunk
    chunk_overlap : int, default=200
        Overlap between consecutive chunks
    metadata : Dict[str, Any], optional
        Additional metadata to include with the document
        
    Returns:
    --------
    Dict[str, Any]
        Indexing results
    """
    await ctx.debug(f"Indexing document from URL: {document_url}")
    
    try:
        # Fetch document
        document_content = await fetch_document(document_url)
        await ctx.debug(f"Fetched document: {len(document_content)} characters")
        
        # Extract text
        text = await extract_text(document_content)
        await ctx.debug(f"Extracted text: {len(text)} characters")
        
        # Create metadata
        if metadata is None:
            metadata = {}
        
        doc_metadata = {
            "source_url": document_url,
            "indexed_at": datetime.now().isoformat(),
            "source_type": "url",
            **metadata
        }
        
        # Process chunks
        result = await chunk_and_process(
            ctx=ctx,
            text=text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            collection=collection,
            metadata=doc_metadata
        )
        
        # Add document-specific info to result
        result["document_url"] = document_url
        result["text_length"] = len(text)
        
        return result
    except Exception as e:
        await ctx.debug(f"Error indexing document: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to index document: {str(e)}"
        }
