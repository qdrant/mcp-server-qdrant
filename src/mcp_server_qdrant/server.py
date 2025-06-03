import os
from mcp_server_qdrant.mcp_server import QdrantMCPServer
from mcp_server_qdrant.settings import (
    EmbeddingProviderSettings,
    QdrantSettings,
    ToolSettings,
)

mcp = QdrantMCPServer(
    tool_settings=ToolSettings(),
    qdrant_settings=QdrantSettings(),
    embedding_provider_settings=EmbeddingProviderSettings(),
    host=os.getenv("FASTMCP_SERVER_HOST", "127.0.0.1"),
    port=int(os.getenv("FASTMCP_SERVER_PORT", "8000")),
)
