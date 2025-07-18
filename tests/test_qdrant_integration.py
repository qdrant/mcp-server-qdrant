import uuid

import pytest

from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider
from mcp_server_qdrant.embeddings.model2vec import Model2VecProvider
from mcp_server_qdrant.qdrant import Entry, QdrantConnector


@pytest.fixture
async def fastembed_embedding_provider():
    """Fixture to provide a FastEmbed embedding provider."""
    return FastEmbedProvider(model_name="sentence-transformers/all-MiniLM-L6-v2")


@pytest.fixture
async def model2vec_embedding_provider():
    """Fixture to provide a Model2Vec embedding provider."""
    return Model2VecProvider(model_name="minishlab/potion-base-8M")


@pytest.fixture
async def qdrant_connector(fastembed_embedding_provider):
    """Fixture to provide a QdrantConnector with in-memory Qdrant client."""
    # Use a random collection name to avoid conflicts between tests
    collection_name = f"test_collection_{uuid.uuid4().hex}"

    # Create connector with in-memory Qdrant
    connector = QdrantConnector(
        qdrant_url=":memory:",
        qdrant_api_key=None,
        collection_name=collection_name,
        embedding_provider=fastembed_embedding_provider,
    )

    yield connector


@pytest.fixture
async def model2vec_qdrant_connector(model2vec_embedding_provider):
    """Fixture to provide a QdrantConnector with Model2Vec embedding provider."""
    # Use a random collection name to avoid conflicts between tests
    collection_name = f"test_m2v_collection_{uuid.uuid4().hex}"

    # Create connector with in-memory Qdrant using Model2Vec provider
    connector = QdrantConnector(
        qdrant_url=":memory:",
        qdrant_api_key=None,
        collection_name=collection_name,
        embedding_provider=model2vec_embedding_provider,
    )

    yield connector


@pytest.mark.asyncio
async def test_fastembed_store_and_search(qdrant_connector):
    """Test storing an entry and then searching for it using FastEmbed embeddings."""
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
async def test_fastembed_search_empty_collection(qdrant_connector):
    """Test searching in an empty collection using FastEmbed embeddings."""
    # Search in an empty collection
    results = await qdrant_connector.search("test query")

    # Verify results
    assert len(results) == 0


@pytest.mark.asyncio
async def test_fastembed_multiple_entries(qdrant_connector):
    """Test storing and searching multiple entries using FastEmbed embeddings."""
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
async def test_fastembed_ensure_collection_exists(qdrant_connector):
    """Test that the collection is created if it doesn't exist using FastEmbed embeddings."""
    # The collection shouldn't exist yet
    assert not await qdrant_connector._client.collection_exists(
        qdrant_connector._default_collection_name
    )

    # Storing an entry should create the collection
    test_entry = Entry(content="Test content")
    await qdrant_connector.store(test_entry)

    # Now the collection should exist
    assert await qdrant_connector._client.collection_exists(
        qdrant_connector._default_collection_name
    )


@pytest.mark.asyncio
async def test_fastembed_metadata_handling(qdrant_connector):
    """Test that metadata is properly stored and retrieved using FastEmbed embeddings."""
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
async def test_fastembed_entry_without_metadata(qdrant_connector):
    """Test storing and retrieving entries without metadata using FastEmbed embeddings."""
    # Store an entry without metadata
    await qdrant_connector.store(Entry(content="Entry without metadata"))

    # Search and verify
    results = await qdrant_connector.search("without metadata")

    assert len(results) == 1
    assert results[0].content == "Entry without metadata"
    assert results[0].metadata is None


@pytest.mark.asyncio
async def test_fastembed_custom_collection_store_and_search(qdrant_connector):
    """Test storing and searching in a custom collection using FastEmbed embeddings."""
    # Define a custom collection name
    custom_collection = f"custom_collection_{uuid.uuid4().hex}"

    # Store a test entry in the custom collection
    test_entry = Entry(
        content="This is stored in a custom collection",
        metadata={"custom": True},
    )
    await qdrant_connector.store(test_entry, collection_name=custom_collection)

    # Search in the custom collection
    results = await qdrant_connector.search(
        "custom collection", collection_name=custom_collection
    )

    # Verify results
    assert len(results) == 1
    assert results[0].content == test_entry.content
    assert results[0].metadata == test_entry.metadata

    # Verify the entry is not in the default collection
    default_results = await qdrant_connector.search("custom collection")
    assert len(default_results) == 0


@pytest.mark.asyncio
async def test_fastembed_multiple_collections(qdrant_connector):
    """Test using multiple collections with the same connector using FastEmbed embeddings."""
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
    entry_default = Entry(content="This belongs to the default collection")

    await qdrant_connector.store(entry_a, collection_name=collection_a)
    await qdrant_connector.store(entry_b, collection_name=collection_b)
    await qdrant_connector.store(entry_default)

    # Search in collection A
    results_a = await qdrant_connector.search("belongs", collection_name=collection_a)
    assert len(results_a) == 1
    assert results_a[0].content == entry_a.content

    # Search in collection B
    results_b = await qdrant_connector.search("belongs", collection_name=collection_b)
    assert len(results_b) == 1
    assert results_b[0].content == entry_b.content

    # Search in default collection
    results_default = await qdrant_connector.search("belongs")
    assert len(results_default) == 1
    assert results_default[0].content == entry_default.content


@pytest.mark.asyncio
async def test_fastembed_nonexistent_collection_search(qdrant_connector):
    """Test searching in a collection that doesn't exist using FastEmbed embeddings."""
    # Search in a collection that doesn't exist
    nonexistent_collection = f"nonexistent_{uuid.uuid4().hex}"
    results = await qdrant_connector.search(
        "test query", collection_name=nonexistent_collection
    )

    # Verify results
    assert len(results) == 0


# Model2Vec Integration Tests with Qdrant

@pytest.mark.asyncio
async def test_model2vec_store_and_search(model2vec_qdrant_connector):
    """Test storing an entry and then searching for it using Model2Vec embeddings."""
    # Store a test entry
    test_entry = Entry(
        content="The quick brown fox jumps over the lazy dog",
        metadata={"source": "test", "importance": "high"},
    )
    await model2vec_qdrant_connector.store(test_entry)

    # Search for the entry
    results = await model2vec_qdrant_connector.search("fox jumps")

    # Verify results
    assert len(results) == 1
    assert results[0].content == test_entry.content
    assert results[0].metadata == test_entry.metadata


@pytest.mark.asyncio
async def test_model2vec_search_empty_collection(model2vec_qdrant_connector):
    """Test searching in an empty collection with Model2Vec embeddings."""
    # Search in an empty collection
    results = await model2vec_qdrant_connector.search("test query")

    # Verify results
    assert len(results) == 0


@pytest.mark.asyncio
async def test_model2vec_multiple_entries(model2vec_qdrant_connector):
    """Test storing and searching multiple entries with Model2Vec embeddings."""
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
        await model2vec_qdrant_connector.store(entry)

    # Search for programming-related entries
    programming_results = await model2vec_qdrant_connector.search("Python programming")
    assert len(programming_results) > 0
    assert any("Python" in result.content for result in programming_results)

    # Search for landmark-related entries
    landmark_results = await model2vec_qdrant_connector.search("Eiffel Tower Paris")
    assert len(landmark_results) > 0
    assert any("Eiffel" in result.content for result in landmark_results)

    # Search for AI-related entries
    ai_results = await model2vec_qdrant_connector.search(
        "artificial intelligence machine learning"
    )
    assert len(ai_results) > 0
    assert any("machine learning" in result.content.lower() for result in ai_results)


@pytest.mark.asyncio
async def test_model2vec_ensure_collection_exists(model2vec_qdrant_connector):
    """Test that the collection is created if it doesn't exist with Model2Vec."""
    # The collection shouldn't exist yet
    assert not await model2vec_qdrant_connector._client.collection_exists(
        model2vec_qdrant_connector._default_collection_name
    )

    # Storing an entry should create the collection
    test_entry = Entry(content="Test content")
    await model2vec_qdrant_connector.store(test_entry)

    # Now the collection should exist
    assert await model2vec_qdrant_connector._client.collection_exists(
        model2vec_qdrant_connector._default_collection_name
    )


@pytest.mark.asyncio
async def test_model2vec_metadata_handling(model2vec_qdrant_connector):
    """Test that metadata is properly stored and retrieved with Model2Vec."""
    # Store entries with different metadata
    metadata1 = {"source": "book", "author": "Jane Doe", "year": 2023}
    metadata2 = {"source": "article", "tags": ["science", "research"]}

    await model2vec_qdrant_connector.store(
        Entry(content="Content with structured metadata", metadata=metadata1)
    )
    await model2vec_qdrant_connector.store(
        Entry(content="Content with list in metadata", metadata=metadata2)
    )

    # Search and verify metadata is preserved
    results = await model2vec_qdrant_connector.search("metadata")

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
async def test_model2vec_entry_without_metadata(model2vec_qdrant_connector):
    """Test storing and retrieving entries without metadata using Model2Vec."""
    # Store an entry without metadata
    await model2vec_qdrant_connector.store(Entry(content="Entry without metadata"))

    # Search and verify
    results = await model2vec_qdrant_connector.search("without metadata")

    assert len(results) == 1
    assert results[0].content == "Entry without metadata"
    assert results[0].metadata is None


@pytest.mark.asyncio
async def test_model2vec_custom_collection_store_and_search(model2vec_qdrant_connector):
    """Test storing and searching in a custom collection with Model2Vec."""
    # Define a custom collection name
    custom_collection = f"custom_collection_{uuid.uuid4().hex}"

    # Store a test entry in the custom collection
    test_entry = Entry(
        content="This is stored in a custom collection",
        metadata={"custom": True},
    )
    await model2vec_qdrant_connector.store(test_entry, collection_name=custom_collection)

    # Search in the custom collection
    results = await model2vec_qdrant_connector.search(
        "custom collection", collection_name=custom_collection
    )

    # Verify results
    assert len(results) == 1
    assert results[0].content == test_entry.content
    assert results[0].metadata == test_entry.metadata

    # Verify the entry is not in the default collection
    default_results = await model2vec_qdrant_connector.search("custom collection")
    assert len(default_results) == 0


@pytest.mark.asyncio
async def test_model2vec_multiple_collections(model2vec_qdrant_connector):
    """Test using multiple collections with the same Model2Vec connector."""
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
    entry_default = Entry(content="This belongs to the default collection")

    await model2vec_qdrant_connector.store(entry_a, collection_name=collection_a)
    await model2vec_qdrant_connector.store(entry_b, collection_name=collection_b)
    await model2vec_qdrant_connector.store(entry_default)

    # Search in collection A
    results_a = await model2vec_qdrant_connector.search("belongs", collection_name=collection_a)
    assert len(results_a) == 1
    assert results_a[0].content == entry_a.content

    # Search in collection B
    results_b = await model2vec_qdrant_connector.search("belongs", collection_name=collection_b)
    assert len(results_b) == 1
    assert results_b[0].content == entry_b.content

    # Search in default collection
    results_default = await model2vec_qdrant_connector.search("belongs")
    assert len(results_default) == 1
    assert results_default[0].content == entry_default.content


@pytest.mark.asyncio
async def test_model2vec_nonexistent_collection_search(model2vec_qdrant_connector):
    """Test searching in a collection that doesn't exist with Model2Vec."""
    # Search in a collection that doesn't exist
    nonexistent_collection = f"nonexistent_{uuid.uuid4().hex}"
    results = await model2vec_qdrant_connector.search(
        "test query", collection_name=nonexistent_collection
    )

    # Verify results
    assert len(results) == 0
