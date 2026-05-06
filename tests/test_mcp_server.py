import pytest
from fastmcp import Client

from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.mcp_server import QdrantMCPServer
from mcp_server_qdrant.settings import QdrantSettings, ToolSettings


class DummyEmbeddingProvider(EmbeddingProvider):
    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return [[0.0] for _ in documents]

    async def embed_query(self, query: str) -> list[float]:
        return [0.0]

    def get_vector_name(self) -> str:
        return "dummy-vector"

    def get_vector_size(self) -> int:
        return 1


class DummyQdrantConnector:
    instances: list["DummyQdrantConnector"] = []

    def __init__(self, *args, **kwargs):
        self.store_calls = []
        DummyQdrantConnector.instances.append(self)

    async def store(self, entry, *, collection_name=None):
        self.store_calls.append((entry, collection_name))

    async def search(self, query, *, collection_name=None, limit=10, query_filter=None):
        return []


@pytest.mark.asyncio
async def test_qdrant_store_allows_omitting_metadata(monkeypatch):
    DummyQdrantConnector.instances.clear()
    monkeypatch.setattr(
        "mcp_server_qdrant.mcp_server.QdrantConnector", DummyQdrantConnector
    )

    server = QdrantMCPServer(
        tool_settings=ToolSettings(),
        qdrant_settings=QdrantSettings(collection_name="memories"),
        embedding_provider=DummyEmbeddingProvider(),
    )

    async with Client(server) as client:
        result = await client.call_tool(
            "qdrant-store",
            {
                "information": "remember this",
            },
        )

    connector = DummyQdrantConnector.instances[0]
    entry, collection_name = connector.store_calls[0]

    assert result.data == "Remembered: remember this in collection memories"
    assert entry.content == "remember this"
    assert entry.metadata is None
    assert collection_name == "memories"
