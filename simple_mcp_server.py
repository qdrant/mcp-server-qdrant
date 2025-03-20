#!/usr/bin/env python
"""
A simplified MCP server for Qdrant
"""
import logging
import asyncio
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, Any

from mcp.server.fastmcp import FastMCP, Context
from mcp_server_qdrant.embeddings.factory import create_embedding_provider
from mcp_server_qdrant.qdrant import QdrantConnector

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("simple-mcp")

# Create a FastMCP instance
fast_mcp = FastMCP("Simple Qdrant MCP")

# Environment variables or defaults
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "claude-test"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

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

# Main function to run the server
if __name__ == "__main__":
    logger.info("Starting Simple Qdrant MCP Server...")
    # Use the synchronous run method which handles the event loop for us
    fast_mcp.run()
