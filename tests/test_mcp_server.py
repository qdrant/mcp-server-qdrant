import pytest

from mcp_server_qdrant.mcp_server import QdrantMCPServer
from mcp_server_qdrant.qdrant import Entry
from mcp_server_qdrant.settings import (
    EmbeddingProviderSettings,
    QdrantSettings,
    ToolSettings,
)


@pytest.fixture
def mcp_server():
    """Fixture to provide a QdrantMCPServer backed by an in-memory client."""
    return QdrantMCPServer(
        tool_settings=ToolSettings(),
        qdrant_settings=QdrantSettings(),
        embedding_provider_settings=EmbeddingProviderSettings(),
    )


def test_format_entry_includes_id(mcp_server):
    """An entry with an id is serialized with its <id> populated."""
    entry = Entry(content="hello", metadata={"k": "v"}, id="point-1")
    formatted = mcp_server.format_entry(entry)
    assert "<id>point-1</id>" in formatted
    assert "<content>hello</content>" in formatted
    assert '<metadata>{"k": "v"}</metadata>' in formatted


def test_format_entry_without_id_raises(mcp_server):
    """Serializing an entry without an id raises instead of emitting an empty <id>."""
    entry = Entry(content="hello", metadata={"k": "v"})
    with pytest.raises(ValueError, match="without an id"):
        mcp_server.format_entry(entry)
