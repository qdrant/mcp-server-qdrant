import uuid

import pytest

from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider
from mcp_server_qdrant.qdrant import Entry, QdrantConnector


@pytest.fixture
async def embedding_provider():
    """Fixture to provide a FastEmbed embedding provider."""
    return FastEmbedProvider(model_name="sentence-transformers/all-MiniLM-L6-v2")


@pytest.fixture
async def qdrant_connector(embedding_provider):
    """Fixture to provide a QdrantConnector with in-memory Qdrant client."""
    # Use a random collection name to avoid conflicts between tests
    collection_name = f"test_collection_{uuid.uuid4().hex}"

    # Create connector with in-memory Qdrant
    connector = QdrantConnector(
        qdrant_url=":memory:",
        qdrant_api_key=None,
        collection_name=collection_name,
        embedding_provider=embedding_provider,
    )

    yield connector


@pytest.mark.asyncio
async def test_store_and_search(qdrant_connector):
    """Test storing an entry and then searching for it."""
    # Store a test entry
    test_entry = Entry(
        content="The quick brown fox jumps over the lazy dog",
        metadata={"source": "test", "importance": "high"},
    )
    await qdrant_connector.store(test_entry)

    # Search for the entry
    results = await qdrant_connector.search("fox jumps")

    # Verify results
    assert len(results) == 1
    assert results[0].content == test_entry.content
    assert results[0].metadata == test_entry.metadata


@pytest.mark.asyncio
async def test_search_empty_collection(qdrant_connector):
    """Test searching in an empty collection."""
    # Search in an empty collection
    results = await qdrant_connector.search("test query")

    # Verify results
    assert len(results) == 0


@pytest.mark.asyncio
async def test_multiple_entries(qdrant_connector):
    """Test storing and searching multiple entries."""
    # Store multiple entries
    entries = [
        Entry(
            content="Python is a programming language",
            metadata={"topic": "programming"},
        ),
        Entry(content="The Eiffel Tower is in Paris", metadata={"topic": "landmarks"}),
        Entry(content="Machine learning is a subset of AI", metadata={"topic": "AI"}),
    ]

    for entry in entries:
        await qdrant_connector.store(entry)

    # Search for programming-related entries
    programming_results = await qdrant_connector.search("Python programming")
    assert len(programming_results) > 0
    assert any("Python" in result.content for result in programming_results)

    # Search for landmark-related entries
    landmark_results = await qdrant_connector.search("Eiffel Tower Paris")
    assert len(landmark_results) > 0
    assert any("Eiffel" in result.content for result in landmark_results)

    # Search for AI-related entries
    ai_results = await qdrant_connector.search(
        "artificial intelligence machine learning"
    )
    assert len(ai_results) > 0
    assert any("machine learning" in result.content.lower() for result in ai_results)


@pytest.mark.asyncio
async def test_ensure_collection_exists(qdrant_connector):
    """Test that the collection is created if it doesn't exist."""
    # The collection shouldn't exist yet
    assert not await qdrant_connector._client.collection_exists(
        qdrant_connector._collection_name
    )

    # Storing an entry should create the collection
    test_entry = Entry(content="Test content")
    await qdrant_connector.store(test_entry)

    # Now the collection should exist
    assert await qdrant_connector._client.collection_exists(
        qdrant_connector._collection_name
    )


@pytest.mark.asyncio
async def test_metadata_handling(qdrant_connector):
    """Test that metadata is properly stored and retrieved."""
    # Store entries with different metadata
    metadata1 = {"source": "book", "author": "Jane Doe", "year": 2023}
    metadata2 = {"source": "article", "tags": ["science", "research"]}

    await qdrant_connector.store(
        Entry(content="Content with structured metadata", metadata=metadata1)
    )
    await qdrant_connector.store(
        Entry(content="Content with list in metadata", metadata=metadata2)
    )

    # Search and verify metadata is preserved
    results = await qdrant_connector.search("metadata")

    assert len(results) == 2

    # Check that both metadata objects are present in the results
    found_metadata1 = False
    found_metadata2 = False

    for result in results:
        if result.metadata.get("source") == "book":
            assert result.metadata.get("author") == "Jane Doe"
            assert result.metadata.get("year") == 2023
            found_metadata1 = True
        elif result.metadata.get("source") == "article":
            assert "science" in result.metadata.get("tags", [])
            assert "research" in result.metadata.get("tags", [])
            found_metadata2 = True

    assert found_metadata1
    assert found_metadata2


@pytest.mark.asyncio
async def test_entry_without_metadata(qdrant_connector):
    """Test storing and retrieving entries without metadata."""
    # Store an entry without metadata
    await qdrant_connector.store(Entry(content="Entry without metadata"))

    # Search and verify
    results = await qdrant_connector.search("without metadata")

    assert len(results) == 1
    assert results[0].content == "Entry without metadata"
    assert results[0].metadata is None
