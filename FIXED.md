# Fixed MCP Server Qdrant

I've fixed a couple of issues in the server code:

1. Updated the code to work with MCP 1.4.1 instead of MCP 0.9.1:
   - Changed from using `@mcp.tool()` to `@fast_mcp.tool()`
   - Created a FastMCP instance for tool registration
   - Updated the server run logic

2. Fixed the Pydantic validation errors:
   - Removed the problematic QdrantSettings usage
   - Created the QdrantConnector directly with the provided parameters

You should now be able to run the server with:

```bash
mcp-server-qdrant
```

This will use the settings from your .env file:
- Collection name: claude-memories
- Qdrant URL: http://localhost:6333
- Embedding model: sentence-transformers/all-MiniLM-L6-v2

Make sure your Qdrant server is running at http://localhost:6333.
