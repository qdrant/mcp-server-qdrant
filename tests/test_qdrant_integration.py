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
    connector = QdrantConnector(
        qdrant_url=":memory:",
        qdrant_api_key=None,
        embedding_provider=embedding_provider,
    )

    yield connector


@pytest.mark.asyncio
async def test_store_and_search(qdrant_connector):
    """Test storing an entry and then searching for it."""
    collection_name = f"test_collection_{uuid.uuid4().hex}"

    # Store a test entry
    test_entry = Entry(
        content="The quick brown fox jumps over the lazy dog",
        metadata={"source": "test", "importance": "high"},
    )
    await qdrant_connector.store(test_entry, collection_name=collection_name)

    # Search for the entry
    results = await qdrant_connector.search(
        "fox jumps", collection_name=collection_name
    )

    # Verify results
    assert len(results) == 1
    assert results[0].content == test_entry.content
    assert results[0].metadata == test_entry.metadata


@pytest.mark.asyncio
async def test_search_empty_collection(qdrant_connector):
    """Test searching in an empty collection."""
    collection_name = f"test_collection_{uuid.uuid4().hex}"

    # Search in an empty collection
    results = await qdrant_connector.search(
        "test query", collection_name=collection_name
    )

    # Verify results
    assert len(results) == 0


@pytest.mark.asyncio
async def test_multiple_entries(qdrant_connector):
    """Test storing and searching multiple entries."""
    collection_name = f"test_collection_{uuid.uuid4().hex}"

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
        await qdrant_connector.store(entry, collection_name=collection_name)

    # Search for programming-related entries
    programming_results = await qdrant_connector.search(
        "Python programming", collection_name=collection_name
    )
    assert len(programming_results) > 0
    assert any("Python" in result.content for result in programming_results)

    # Search for landmark-related entries
    landmark_results = await qdrant_connector.search(
        "Eiffel Tower Paris", collection_name=collection_name
    )
    assert len(landmark_results) > 0
    assert any("Eiffel" in result.content for result in landmark_results)

    # Search for AI-related entries
    ai_results = await qdrant_connector.search(
        "artificial intelligence machine learning", collection_name=collection_name
    )
    assert len(ai_results) > 0
    assert any("machine learning" in result.content.lower() for result in ai_results)


@pytest.mark.asyncio
async def test_ensure_collection_exists(qdrant_connector):
    """Test that the collection is created if it doesn't exist."""
    collection_name = f"test_collection_{uuid.uuid4().hex}"

    # The collection shouldn't exist yet
    assert not await qdrant_connector._client.collection_exists(collection_name)

    # Storing an entry should create the collection
    test_entry = Entry(content="Test content")
    await qdrant_connector.store(test_entry, collection_name=collection_name)

    # Now the collection should exist
    assert await qdrant_connector._client.collection_exists(collection_name)


@pytest.mark.asyncio
async def test_metadata_handling(qdrant_connector):
    """Test that metadata is properly stored and retrieved."""
    collection_name = f"test_collection_{uuid.uuid4().hex}"

    # Store entries with different metadata
    metadata1 = {"source": "book", "author": "Jane Doe", "year": 2023}
    metadata2 = {"source": "article", "tags": ["science", "research"]}

    await qdrant_connector.store(
        Entry(content="Content with structured metadata", metadata=metadata1),
        collection_name=collection_name,
    )
    await qdrant_connector.store(
        Entry(content="Content with list in metadata", metadata=metadata2),
        collection_name=collection_name,
    )

    # Search and verify metadata is preserved
    results = await qdrant_connector.search("metadata", collection_name=collection_name)

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
    collection_name = f"test_collection_{uuid.uuid4().hex}"

    # Store an entry without metadata
    await qdrant_connector.store(
        Entry(content="Entry without metadata"), collection_name=collection_name
    )

    # Search and verify
    results = await qdrant_connector.search(
        "without metadata", collection_name=collection_name
    )

    assert len(results) == 1
    assert results[0].content == "Entry without metadata"
    assert results[0].metadata is None


@pytest.mark.asyncio
async def test_multiple_collections(qdrant_connector):
    """Test using multiple collections with the same connector."""
    # Define two custom collection names
    collection_a = f"collection_a_{uuid.uuid4().hex}"
    collection_b = f"collection_b_{uuid.uuid4().hex}"

    # Store entries in different collections
    entry_a = Entry(
        content="This belongs to collection A", metadata={"collection": "A"}
    )
    entry_b = Entry(
        content="This belongs to collection B", metadata={"collection": "B"}
    )

    await qdrant_connector.store(entry_a, collection_name=collection_a)
    await qdrant_connector.store(entry_b, collection_name=collection_b)

    # Search in collection A
    results_a = await qdrant_connector.search("belongs", collection_name=collection_a)
    assert len(results_a) == 1
    assert results_a[0].content == entry_a.content

    # Search in collection B
    results_b = await qdrant_connector.search("belongs", collection_name=collection_b)
    assert len(results_b) == 1
    assert results_b[0].content == entry_b.content


@pytest.mark.asyncio
async def test_nonexistent_collection_search(qdrant_connector):
    """Test searching in a collection that doesn't exist."""
    # Search in a collection that doesn't exist
    nonexistent_collection = f"nonexistent_{uuid.uuid4().hex}"
    results = await qdrant_connector.search(
        "test query", collection_name=nonexistent_collection
    )

    # Verify results
    assert len(results) == 0
