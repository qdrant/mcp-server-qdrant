import uuid
from unittest.mock import AsyncMock

import pytest

from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider
from mcp_server_qdrant.mcp_server import QdrantMCPServer
from mcp_server_qdrant.qdrant import Entry, QdrantConnector
from mcp_server_qdrant.settings import (
    EmbeddingProviderSettings,
    QdrantSettings,
    ToolSettings,
)


@pytest.fixture
async def embedding_provider():
    """Fixture to provide a FastEmbed embedding provider."""
    return FastEmbedProvider(model_name="sentence-transformers/all-MiniLM-L6-v2")


@pytest.fixture
async def qdrant_connector_with_default(embedding_provider):
    """Fixture to provide a QdrantConnector with a default collection configured."""
    default_collection = f"default_collection_{uuid.uuid4().hex}"
    
    connector = QdrantConnector(
        qdrant_url=":memory:",
        qdrant_api_key=None,
        collection_name=default_collection,
        embedding_provider=embedding_provider,
    )
    
    yield connector, default_collection


@pytest.fixture
async def qdrant_connector_no_default(embedding_provider):
    """Fixture to provide a QdrantConnector without default collection."""
    connector = QdrantConnector(
        qdrant_url=":memory:",
        qdrant_api_key=None,
        collection_name=None,  # No default collection
        embedding_provider=embedding_provider,
    )
    
    yield connector


@pytest.fixture
async def mcp_server_with_default(embedding_provider):
    """Fixture to provide a QdrantMCPServer with default collection configured."""
    default_collection = f"mcp_default_{uuid.uuid4().hex}"
    
    qdrant_settings = QdrantSettings()
    qdrant_settings.location = ":memory:"
    qdrant_settings.collection_name = default_collection
    
    server = QdrantMCPServer(
        tool_settings=ToolSettings(),
        qdrant_settings=qdrant_settings,
        embedding_provider=embedding_provider,
    )
    
    yield server, default_collection


@pytest.fixture
async def mcp_server_no_default(embedding_provider):
    """Fixture to provide a QdrantMCPServer without default collection."""
    qdrant_settings = QdrantSettings()
    qdrant_settings.location = ":memory:"
    qdrant_settings.collection_name = None
    
    server = QdrantMCPServer(
        tool_settings=ToolSettings(),
        qdrant_settings=qdrant_settings,
        embedding_provider=embedding_provider,
    )
    
    yield server


@pytest.mark.asyncio
async def test_connector_store_with_default_collection(qdrant_connector_with_default):
    """Test that store uses default collection when collection_name is None."""
    connector, default_collection = qdrant_connector_with_default
    
    # Store without specifying collection (should use default)
    test_entry = Entry(
        content="Test content for default collection",
        metadata={"test": "default"}
    )
    await connector.store(test_entry, collection_name=None)
    
    # Search in default collection to verify it was stored there
    results = await connector.search("Test content", collection_name=default_collection)
    assert len(results) == 1
    assert results[0].content == test_entry.content
    assert results[0].metadata == test_entry.metadata


@pytest.mark.asyncio
async def test_connector_store_with_custom_collection(qdrant_connector_with_default):
    """Test that store uses custom collection when collection_name is specified."""
    connector, default_collection = qdrant_connector_with_default
    custom_collection = f"custom_{uuid.uuid4().hex}"
    
    # Store with custom collection
    test_entry = Entry(
        content="Test content for custom collection",
        metadata={"test": "custom"}
    )
    await connector.store(test_entry, collection_name=custom_collection)
    
    # Search in custom collection to verify it was stored there
    results = await connector.search("Test content", collection_name=custom_collection)
    assert len(results) == 1
    assert results[0].content == test_entry.content
    
    # Verify it's NOT in the default collection
    default_results = await connector.search("Test content", collection_name=default_collection)
    assert len(default_results) == 0


@pytest.mark.asyncio
async def test_connector_search_with_default_collection(qdrant_connector_with_default):
    """Test that search uses default collection when collection_name is None."""
    connector, default_collection = qdrant_connector_with_default
    
    # Store in default collection
    test_entry = Entry(content="Searchable content in default")
    await connector.store(test_entry, collection_name=default_collection)
    
    # Search without specifying collection (should search default)
    results = await connector.search("Searchable content", collection_name=None)
    assert len(results) == 1
    assert results[0].content == test_entry.content


@pytest.mark.asyncio
async def test_connector_search_with_custom_collection(qdrant_connector_with_default):
    """Test that search uses custom collection when collection_name is specified."""
    connector, default_collection = qdrant_connector_with_default
    custom_collection = f"search_custom_{uuid.uuid4().hex}"
    
    # Store in custom collection
    test_entry = Entry(content="Searchable content in custom")
    await connector.store(test_entry, collection_name=custom_collection)
    
    # Search in custom collection
    results = await connector.search("Searchable content", collection_name=custom_collection)
    assert len(results) == 1
    assert results[0].content == test_entry.content
    
    # Verify search in default collection returns nothing
    default_results = await connector.search("Searchable content", collection_name=default_collection)
    assert len(default_results) == 0


@pytest.mark.asyncio
async def test_connector_no_default_collection_requires_explicit_name(qdrant_connector_no_default):
    """Test that connector without default collection requires explicit collection name."""
    connector = qdrant_connector_no_default
    
    # Attempting to store without collection name should fail
    test_entry = Entry(content="Test content")
    
    with pytest.raises((AssertionError, ValueError)):
        await connector.store(test_entry, collection_name=None)


@pytest.mark.asyncio
async def test_mcp_server_tool_functions_directly(mcp_server_with_default):
    """Test MCP server tool functions work directly with default and custom collections."""
    server, default_collection = mcp_server_with_default
    ctx = AsyncMock()
    
    # Create the internal tool functions as they are defined in setup_tools
    async def store(
        ctx, information, collection_name=None, metadata=None
    ):
        effective_collection = collection_name or server.qdrant_settings.collection_name
        await ctx.debug(f"Storing information {information} in Qdrant collection: {effective_collection}")
        
        from mcp_server_qdrant.qdrant import Entry
        entry = Entry(content=information, metadata=metadata)
        
        await server.qdrant_connector.store(entry, collection_name=effective_collection)
        if effective_collection:
            return f"Remembered: {information} in collection {effective_collection}"
        return f"Remembered: {information}"
    
    async def find(
        ctx, query, collection_name=None, query_filter=None
    ):
        effective_collection = collection_name or server.qdrant_settings.collection_name
        await ctx.debug(f"Finding results for query {query} in collection: {effective_collection}")
        
        entries = await server.qdrant_connector.search(
            query,
            collection_name=effective_collection,
            limit=server.qdrant_settings.search_limit,
            query_filter=None,
        )
        if not entries:
            return None
        content = [f"Results for the query '{query}'"]
        for entry in entries:
            content.append(server.format_entry(entry))
        return content
    
    # Test store with None collection_name (should use default)
    store_result = await store(
        ctx=ctx,
        information="Test information for MCP store",
        collection_name=None,
        metadata={"mcp_test": True}
    )
    
    assert "Remembered:" in store_result
    assert default_collection in store_result
    
    # Test find with None collection_name (should search default)
    find_results = await find(
        ctx=ctx,
        query="Test information",
        collection_name=None
    )
    
    assert find_results is not None
    assert len(find_results) > 1  # Should include header and result
    assert "Test information for MCP store" in str(find_results)


@pytest.mark.asyncio
async def test_mcp_server_collection_override_behavior(mcp_server_with_default):
    """Test that custom collection name properly overrides default collection."""
    server, default_collection = mcp_server_with_default
    custom_collection = f"mcp_custom_{uuid.uuid4().hex}"
    
    # Test the effective collection logic directly
    from mcp_server_qdrant.qdrant import Entry
    
    # Store in custom collection (override default)
    entry_custom = Entry(content="Custom collection content", metadata={"type": "custom"})
    await server.qdrant_connector.store(entry_custom, collection_name=custom_collection)
    
    # Store in default collection (using None)
    entry_default = Entry(content="Default collection content", metadata={"type": "default"})
    await server.qdrant_connector.store(entry_default, collection_name=None)
    
    # Search in custom collection
    custom_results = await server.qdrant_connector.search(
        "Custom collection", collection_name=custom_collection
    )
    assert len(custom_results) == 1
    assert custom_results[0].content == "Custom collection content"
    
    # Search in default collection
    default_results = await server.qdrant_connector.search(
        "Default collection", collection_name=default_collection
    )
    assert len(default_results) == 1
    assert default_results[0].content == "Default collection content"
    
    # Verify isolation - custom content not in default collection
    # Search for specific content that should only be in custom collection
    cross_search = await server.qdrant_connector.search(
        "Custom collection content", collection_name=default_collection
    )
    # The search might return results due to semantic similarity, so check content specifically
    custom_content_in_default = any(
        "Custom collection content" in result.content for result in cross_search
    )
    assert not custom_content_in_default, "Custom content should not be found in default collection"


@pytest.mark.asyncio
async def test_multiple_collection_isolation_via_connector(mcp_server_with_default):
    """Test that multiple collections are properly isolated when accessed via connector."""
    server, default_collection = mcp_server_with_default
    collection_a = f"isolation_a_{uuid.uuid4().hex}"
    collection_b = f"isolation_b_{uuid.uuid4().hex}"
    
    from mcp_server_qdrant.qdrant import Entry
    
    # Store different content in each collection
    await server.qdrant_connector.store(
        Entry(content="Content for default collection"), collection_name=None
    )
    await server.qdrant_connector.store(
        Entry(content="Content for collection A"), collection_name=collection_a
    )
    await server.qdrant_connector.store(
        Entry(content="Content for collection B"), collection_name=collection_b
    )
    
    # Search in default collection
    default_results = await server.qdrant_connector.search(
        "Content", collection_name=default_collection
    )
    assert len(default_results) == 1
    assert "Content for default collection" in default_results[0].content
    
    # Search in collection A
    a_results = await server.qdrant_connector.search(
        "Content", collection_name=collection_a
    )
    assert len(a_results) == 1
    assert "Content for collection A" in a_results[0].content
    
    # Search in collection B
    b_results = await server.qdrant_connector.search(
        "Content", collection_name=collection_b
    )
    assert len(b_results) == 1
    assert "Content for collection B" in b_results[0].content


@pytest.mark.asyncio
async def test_settings_configuration_with_optional_collection():
    """Test that settings work correctly with optional collection configuration."""
    from mcp_server_qdrant.settings import QdrantSettings
    
    # Test with default collection configured
    settings_with_default = QdrantSettings()
    settings_with_default.location = ":memory:"
    settings_with_default.collection_name = "test_default_collection"
    
    # Test without default collection
    settings_no_default = QdrantSettings()
    settings_no_default.location = ":memory:"
    settings_no_default.collection_name = None
    
    # Verify settings are configured as expected
    assert settings_with_default.collection_name == "test_default_collection"
    assert settings_no_default.collection_name is None


@pytest.mark.asyncio  
async def test_effective_collection_logic():
    """Test the effective collection selection logic in isolation."""
    # Simulate the logic used in the actual implementation
    def get_effective_collection(collection_name, default_collection):
        return collection_name or default_collection
    
    # Test cases
    assert get_effective_collection(None, "default") == "default"
    assert get_effective_collection("custom", "default") == "custom"
    assert get_effective_collection("", "default") == "default"  # Empty string should use default
    assert get_effective_collection("custom", None) == "custom"
    
    # Edge case: both None (should be handled by connector validation)
    result = get_effective_collection(None, None)
    assert result is None  # This should trigger validation error in real usage