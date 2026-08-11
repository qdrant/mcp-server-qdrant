import pytest

from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider
from mcp_server_qdrant.qdrant import Entry, QdrantConnector
from mcp_server_qdrant.settings import QdrantSettings


def test_allowed_collections_parses_csv(monkeypatch):
    monkeypatch.setenv("QDRANT_ALLOW_COLLECTIONS", "notes, docs ,memories")
    assert QdrantSettings().allowed_collections() == {"notes", "docs", "memories"}


def test_allowed_collections_none_when_unset(monkeypatch):
    monkeypatch.delenv("QDRANT_ALLOW_COLLECTIONS", raising=False)
    assert QdrantSettings().allowed_collections() is None


@pytest.fixture
async def provider():
    return FastEmbedProvider(model_name="sentence-transformers/all-MiniLM-L6-v2")


def _connector(provider, allow):
    return QdrantConnector(
        qdrant_url=":memory:",
        qdrant_api_key=None,
        collection_name=None,
        embedding_provider=provider,
        allow_collections=allow,
    )


@pytest.mark.asyncio
async def test_disallowed_collection_is_rejected(provider):
    connector = _connector(provider, {"allowed"})
    with pytest.raises(ValueError, match="not allowed"):
        await connector.store(Entry(content="x"), collection_name="secret")
    with pytest.raises(ValueError, match="not allowed"):
        await connector.search("x", collection_name="secret")


@pytest.mark.asyncio
async def test_allowed_collection_works(provider):
    connector = _connector(provider, {"allowed"})
    await connector.store(Entry(content="hello world"), collection_name="allowed")
    results = await connector.search("hello", collection_name="allowed")
    assert any("hello" in e.content for e in results)


@pytest.mark.asyncio
async def test_no_allowlist_permits_any_collection(provider):
    connector = _connector(provider, None)
    await connector.store(Entry(content="anything"), collection_name="whatever")
    results = await connector.search("anything", collection_name="whatever")
    assert len(results) >= 1
