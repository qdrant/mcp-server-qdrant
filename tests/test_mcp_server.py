import inspect

import pytest

from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.mcp_server import QdrantMCPServer
from mcp_server_qdrant.settings import QdrantSettings, ToolSettings


class FakeEmbeddingProvider(EmbeddingProvider):
    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return [[1.0] for _ in documents]

    async def embed_query(self, query: str) -> list[float]:
        return [1.0]

    def get_vector_name(self) -> str:
        return "fake"

    def get_vector_size(self) -> int:
        return 1


@pytest.fixture
def tool_settings():
    return ToolSettings()


@pytest.fixture
def embedding_provider():
    return FakeEmbeddingProvider()


async def create_tools(tool_settings, embedding_provider) -> dict:
    server = QdrantMCPServer(
        tool_settings=tool_settings,
        qdrant_settings=QdrantSettings(),
        embedding_provider=embedding_provider,
    )
    return await server.get_tools()


@pytest.mark.asyncio
async def test_read_only_mode_only_exposes_find_tool(
    monkeypatch, tool_settings, embedding_provider
):
    """Test that write tools are hidden in read-only mode."""
    monkeypatch.setenv("QDRANT_READ_ONLY", "1")

    tools = await create_tools(tool_settings, embedding_provider)

    assert set(tools.keys()) == {"qdrant-find"}


@pytest.mark.asyncio
async def test_collection_name_default_removes_collection_name_parameter(
    monkeypatch, tool_settings, embedding_provider
):
    """Test that default collection binding removes collection_name from tool signatures."""
    monkeypatch.setenv("COLLECTION_NAME", "memories")

    tools = await create_tools(tool_settings, embedding_provider)

    assert "collection_name" not in inspect.signature(tools["qdrant-find"].fn).parameters
    assert "collection_name" not in inspect.signature(tools["qdrant-store"].fn).parameters
    assert "collection_name" not in inspect.signature(tools["qdrant-edit"].fn).parameters


@pytest.mark.asyncio
async def test_without_default_collection_collection_name_parameter_is_required(
    monkeypatch, tool_settings, embedding_provider
):
    """Test that collection_name remains exposed when no default collection is configured."""
    monkeypatch.delenv("COLLECTION_NAME", raising=False)

    tools = await create_tools(tool_settings, embedding_provider)

    assert "collection_name" in inspect.signature(tools["qdrant-find"].fn).parameters
    assert "collection_name" in inspect.signature(tools["qdrant-store"].fn).parameters
    assert "collection_name" in inspect.signature(tools["qdrant-edit"].fn).parameters
