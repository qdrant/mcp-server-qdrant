import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any

from mcp_server_qdrant.mcp_server import QdrantMCPServer
from mcp_server_qdrant.qdrant import Entry

@pytest.fixture
def mock_server():
    """Instantiates a mock server and captures tools using a pre-emptive monkeypatch."""
    # 1. Setup mock settings to satisfy the constructor
    mock_settings = MagicMock()
    mock_settings.search_limit = 10
    mock_settings.location = ":memory:"
    mock_settings.url = mock_settings.host = mock_settings.path = mock_settings.local_path = None
    
    #Force the server into write-mode so the store tool registers
    mock_settings.read_only = False

    mock_tool_settings = MagicMock()
    mock_tool_settings.tool_find_description = "Find memories."
    mock_tool_settings.tool_store_description = "Store memories."
    
    captured_tools = {}

    # 2. Define the interception wrapper
    # We use *args, **kwargs to be completely blind to how FastMCP passes data
    original_tool_method = QdrantMCPServer.tool

    def capture_tool_wrapper(self, *args, **kwargs):
        # Capture the function (usually first arg) and the name (keyword or second arg)
        fn = args[0] if args else kwargs.get('fn')
        name = kwargs.get('name')
        if not name and len(args) > 1:
            name = args[1]
        
        if fn:
            # If no name is found, fallback to the function's own name
            key = name or getattr(fn, "__name__", "unknown")
            captured_tools[key] = fn
            
        return original_tool_method(self, *args, **kwargs)

    # 3. Apply the patch to the CLASS, then instantiate
    with patch.object(QdrantMCPServer, 'tool', capture_tool_wrapper):
        server = QdrantMCPServer(
            qdrant_settings=mock_settings,
            tool_settings=mock_tool_settings,
            embedding_provider=MagicMock()
        )
    
    # 4. Mock the database connector AFTER init is finished
    server.qdrant_connector = MagicMock()
    server.qdrant_connector.search = AsyncMock()
    server.qdrant_connector.store = AsyncMock()
    
    # Normalize tool names (remove 'qdrant-' prefix if present)
    normalized = {}
    for name, fn in captured_tools.items():
        clean_name = name.replace('qdrant-', '')
        normalized[clean_name] = fn
    captured_tools.update(normalized)
    
    server.captured_tools = captured_tools
    return server

@pytest.mark.asyncio
async def test_store_requires_confirmation(mock_server):
    """Verify that autonomous agents cannot write data without the confirm_store flag."""
    store_fn = mock_server.captured_tools.get('store')
    assert store_fn is not None, f"Store tool not found. Registered: {list(mock_server.captured_tools.keys())}"
    
    mock_ctx = MagicMock()
    mock_ctx.debug = AsyncMock()
    
    # Execute WITHOUT confirmation flag
    # These handlers are closures that already have 'self' bound, so we don't pass it.
    result = await store_fn(
        ctx=mock_ctx,
        information="Malicious hallucinated data",
        collection_name="test_collection",
        confirm_store=False
    )
    
    assert "Error: Storage operation cancelled" in result
    mock_server.qdrant_connector.store.assert_not_called()

@pytest.mark.asyncio
async def test_find_metadata_firewall(mock_server):
    """Verify that the search tool strips unrequested PII from the metadata payload."""
    find_fn = mock_server.captured_tools.get('find')
    assert find_fn is not None, f"Find tool not found. Registered: {list(mock_server.captured_tools.keys())}"
    
    mock_ctx = MagicMock()
    mock_ctx.debug = AsyncMock()
    
    # Mock database returning a record with sensitive PII
    mock_server.qdrant_connector.search.return_value = [
        Entry(
            content="Internal document",
            metadata={"public": "ok", "ssn": "secret-PII", "internal_note": "top secret"}
        )
    ]
    
    # Execute search with a specific whitelist
    results = await find_fn(
        ctx=mock_ctx,
        query="test",
        collection_name="test_collection",
        selected_metadata_keys=["public"]
    )
    
    assert results is not None, "Expected search results, but got None" # Add this line
    output_string = "".join(results)
    assert "public" in output_string
    assert "ssn" not in output_string
    assert "internal_note" not in output_string