import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from mcp_server_qdrant.mcp_server import QdrantMCPServer, describe_exception
from mcp_server_qdrant.settings import (
    EmbeddingProviderSettings,
    QdrantSettings,
    ToolSettings,
)


@pytest.fixture
def mcp_server():
    """Fixture providing a QdrantMCPServer backed by an in-memory Qdrant."""
    return QdrantMCPServer(
        tool_settings=ToolSettings(),
        qdrant_settings=QdrantSettings(),
        embedding_provider_settings=EmbeddingProviderSettings(),
    )


class TestDescribeException:
    def test_uses_message_when_present(self):
        assert describe_exception(ValueError("boom")) == "ValueError: boom"

    def test_falls_back_to_class_name_for_empty_message(self):
        # A bare AssertionError has an empty string representation.
        assert describe_exception(AssertionError()) == "AssertionError"

    def test_strips_whitespace_only_messages(self):
        assert describe_exception(RuntimeError("   ")) == "RuntimeError"


class TestToolErrorSurfacing:
    """Regression tests for issue #151: tools returned empty errors."""

    @pytest.mark.asyncio
    async def test_find_surfaces_non_empty_error(self, mcp_server, monkeypatch):
        # Simulate a backend failure whose string representation is empty.
        async def boom(*args, **kwargs):
            raise AssertionError()

        monkeypatch.setattr(mcp_server.qdrant_connector, "search", boom)

        async with Client(mcp_server) as client:
            with pytest.raises(ToolError) as exc_info:
                await client.call_tool(
                    "qdrant-find",
                    {"query": "anything", "collection_name": "some-collection"},
                )

        message = str(exc_info.value)
        # Before the fix this was "Error calling tool 'qdrant-find': " with an
        # empty tail. The real error must now propagate to the client.
        assert message.strip() != ""
        assert "qdrant-find failed" in message
        assert "AssertionError" in message

    @pytest.mark.asyncio
    async def test_store_surfaces_non_empty_error(self, mcp_server, monkeypatch):
        async def boom(*args, **kwargs):
            raise AssertionError()

        monkeypatch.setattr(mcp_server.qdrant_connector, "store", boom)

        async with Client(mcp_server) as client:
            with pytest.raises(ToolError) as exc_info:
                await client.call_tool(
                    "qdrant-store",
                    {"information": "hello", "collection_name": "some-collection"},
                )

        message = str(exc_info.value)
        assert message.strip() != ""
        assert "qdrant-store failed" in message
        assert "AssertionError" in message

    @pytest.mark.asyncio
    async def test_find_propagates_real_message(self, mcp_server, monkeypatch):
        async def boom(*args, **kwargs):
            raise RuntimeError("connection refused")

        monkeypatch.setattr(mcp_server.qdrant_connector, "search", boom)

        async with Client(mcp_server) as client:
            with pytest.raises(ToolError) as exc_info:
                await client.call_tool(
                    "qdrant-find",
                    {"query": "anything", "collection_name": "some-collection"},
                )

        assert "connection refused" in str(exc_info.value)
