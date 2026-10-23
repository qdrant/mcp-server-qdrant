import uuid

import pytest
from fastmcp import Client

from mcp_server_qdrant.mcp_server import QdrantMCPServer
from mcp_server_qdrant.settings import (
    EmbeddingProviderSettings,
    QdrantSettings,
    ToolSettings,
)


def make_server(monkeypatch, **env) -> QdrantMCPServer:
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return QdrantMCPServer(
        tool_settings=ToolSettings(),
        qdrant_settings=QdrantSettings(),
        embedding_provider_settings=EmbeddingProviderSettings(),
    )


@pytest.mark.asyncio
async def test_list_collections(monkeypatch):
    """Test that the tool lists the collections created by storing entries."""
    server = make_server(monkeypatch)
    names = [f"test_collection_{uuid.uuid4().hex}" for _ in range(2)]

    async with Client(server) as client:
        for name in names:
            await client.call_tool(
                "qdrant-store", {"information": "test", "collection_name": name}
            )
        result = await client.call_tool("qdrant-list-collections", {})

    listed = "".join(content.text for content in result)
    for name in names:
        assert name in listed


@pytest.mark.asyncio
async def test_list_collections_read_only(monkeypatch):
    """Test that the tool is available in read-only mode."""
    server = make_server(monkeypatch, QDRANT_READ_ONLY="1")

    async with Client(server) as client:
        tools = {tool.name for tool in await client.list_tools()}

    assert "qdrant-list-collections" in tools
    assert "qdrant-store" not in tools


@pytest.mark.asyncio
async def test_list_collections_disabled_with_default_collection(monkeypatch):
    """Test that the tool is not exposed when a default collection is set."""
    server = make_server(monkeypatch, COLLECTION_NAME="default")

    async with Client(server) as client:
        tools = {tool.name for tool in await client.list_tools()}

    assert "qdrant-list-collections" not in tools
