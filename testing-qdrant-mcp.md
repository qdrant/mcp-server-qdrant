# Testing Guide for Enhanced Qdrant MCP Server

This guide provides a comprehensive approach to testing the enhanced Qdrant MCP server with automated tests and quality assurance workflows.

## Table of Contents

- [Testing Strategy Overview](#testing-strategy-overview)
- [Test Environment Setup](#test-environment-setup)
- [Automated Testing Framework](#automated-testing-framework)
- [Unit Tests](#unit-tests)
- [Integration Tests](#integration-tests)
- [End-to-End Tests](#end-to-end-tests)
- [Performance Tests](#performance-tests)
- [Test Automation and CI/CD](#test-automation-and-cicd)
- [Manual Testing Guide](#manual-testing-guide)
- [Test Data Generation](#test-data-generation)
- [Common Testing Issues](#common-testing-issues)

## Testing Strategy Overview

The testing strategy for the Qdrant MCP server follows a layered approach:

1. **Unit Tests**: Verify that individual functions and classes work correctly in isolation
2. **Integration Tests**: Test interactions between components and with Qdrant
3. **End-to-End Tests**: Ensure the complete MCP server functions as expected
4. **Performance Tests**: Verify that the server meets performance requirements

Each layer provides different insights and coverage, creating a comprehensive test suite.

## Test Environment Setup

### Development Environment

Create a standardized test environment to ensure consistent results:

```python
# tests/conftest.py
import os
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from qdrant_client import QdrantClient
from mcp.server.fastmcp import Context

@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def mock_qdrant_client():
    """Create a mock Qdrant client for testing."""
    client = AsyncMock(spec=QdrantClient)
    # Set up common mock behaviors
    client.search.return_value = []
    client.get_collection.return_value = {"vectors_count": 100}
    return client

@pytest.fixture
async def mock_context():
    """Create a mock MCP context for testing."""
    ctx = AsyncMock(spec=Context)
    ctx.debug = AsyncMock()
    ctx.request_context = Mock()
    ctx.request_context.lifespan_context = {
        "qdrant_connector": Mock(),
    }
    ctx.request_context.lifespan_context["qdrant_connector"]._client = AsyncMock(spec=QdrantClient)
    ctx.request_context.lifespan_context["qdrant_connector"]._embedding_provider = AsyncMock()
    ctx.request_context.lifespan_context["qdrant_connector"]._embedding_provider.embed_query.return_value = [0.1] * 384
    ctx.request_context.lifespan_context["qdrant_connector"]._embedding_provider.get_vector_name.return_value = "default"
    return ctx

@pytest.fixture
async def test_qdrant_client():
    """Create a real Qdrant client connected to a test instance."""
    # Use in-memory Qdrant for testing
    client = QdrantClient(":memory:")
    yield client
    # Clean up collections after the test
    try:
        collections = await client.get_collections()
        for collection in collections.collections:
            await client.delete_collection(collection.name)
    except Exception:
        pass
```

### Integration Test Environment

```python
# tests/integration/conftest.py
import os
import pytest
import tempfile
from qdrant_client import QdrantClient
from mcp_server_qdrant.qdrant import QdrantConnector
from mcp_server_qdrant.embeddings.factory import create_embedding_provider
from mcp_server_qdrant.embeddings.types import EmbeddingProviderType

@pytest.fixture
async def test_embedding_provider():
    """Create a real embedding provider for testing."""
    return create_embedding_provider({
        "provider_type": EmbeddingProviderType.FASTEMBED,
        "model_name": "sentence-transformers/all-MiniLM-L6-v2"
    })

@pytest.fixture
async def test_qdrant_connector(test_embedding_provider):
    """Create a QdrantConnector for integration testing."""
    # Use a temporary path for Qdrant storage
    temp_dir = tempfile.TemporaryDirectory()
    
    connector = QdrantConnector(
        location=None,
        api_key=None,
        collection_name="test_collection",
        embedding_provider=test_embedding_provider,
        local_path=temp_dir.name
    )
    
    yield connector
    
    # Clean up
    temp_dir.cleanup()
```

## Automated Testing Framework

The testing framework is built on pytest with the following enhancements:

1. **pytest-asyncio**: For testing asynchronous functions
2. **pytest-cov**: For measuring code coverage
3. **pytest-xdist**: For parallel test execution

### Directory Structure

```
/tests
  ├── __init__.py
  ├── conftest.py               # Global fixtures
  ├── unit/                     # Unit tests
  │   ├── __init__.py
  │   ├── test_search.py
  │   ├── test_collection_mgmt.py
  │   ├── test_data_processing.py
  │   ├── test_advanced_query.py
  │   ├── test_analytics.py
  │   └── test_document_processing.py
  ├── integration/              # Integration tests
  │   ├── __init__.py
  │   ├── conftest.py           # Integration-specific fixtures
  │   ├── test_search_integration.py
  │   └── ...
  ├── e2e/                      # End-to-end tests
  │   ├── __init__.py
  │   ├── conftest.py           # E2E-specific fixtures
  │   ├── test_mcp_server.py
  │   └── ...
  └── performance/              # Performance tests
      ├── __init__.py
      ├── test_search_performance.py
      └── ...
```

### Test Configuration

Create a `pytest.ini` file in your project root:

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
asyncio_mode = auto
addopts = --cov=src/mcp_server_qdrant --cov-report=term --cov-report=html
markers =
    unit: mark a test as a unit test
    integration: mark a test as an integration test
    e2e: mark a test as an end-to-end test
    performance: mark a test as a performance test
```

## Unit Tests

Unit tests focus on testing individual functions in isolation, using mocks for dependencies.

### Example Unit Test for Natural Language Query

```python
# tests/unit/test_search.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from mcp_server_qdrant.tools.search.nlq import natural_language_query

@pytest.mark.unit
async def test_natural_language_query(mock_context):
    """Test that the natural language query function works correctly."""
    # Arrange
    query = "test query"
    collection = "test_collection"
    limit = 5
    
    mock_client = mock_context.request_context.lifespan_context["qdrant_connector"]._client
    mock_client.search.return_value = [
        MagicMock(
            id="1",
            score=0.9,
            payload={"document": "Document 1", "metadata": {"source": "test"}}
        ),
        MagicMock(
            id="2",
            score=0.8,
            payload={"document": "Document 2", "metadata": {"source": "test"}}
        )
    ]
    
    # Act
    result = await natural_language_query(
        ctx=mock_context,
        query=query,
        collection=collection,
        limit=limit
    )
    
    # Assert
    assert len(result) == 2
    assert result[0]["score"] == 0.9
    assert result[0]["document"] == "Document 1"
    assert result[1]["document"] == "Document 2"
    mock_client.search.assert_called_once()
    search_args = mock_client.search.call_args[1]
    assert search_args["collection_name"] == collection
    assert search_args["limit"] == limit
```

### Example Unit Test for Collection Creation

```python
# tests/unit/test_collection_mgmt.py
import pytest
from unittest.mock import AsyncMock
from qdrant_client.http.models import VectorParams, Distance
from mcp_server_qdrant.tools.collection_mgmt.create_collection import create_collection

@pytest.mark.unit
async def test_create_collection(mock_context):
    """Test that the create collection function works correctly."""
    # Arrange
    name = "test_collection"
    vector_size = 384
    distance = "Cosine"
    
    mock_client = mock_context.request_context.lifespan_context["qdrant_connector"]._client
    mock_client.create_collection.return_value = None
    
    # Act
    result = await create_collection(
        ctx=mock_context,
        name=name,
        vector_size=vector_size,
        distance=distance
    )
    
    # Assert
    assert result["status"] == "success"
    assert result["collection_name"] == name
    mock_client.create_collection.assert_called_once()
    call_args = mock_client.create_collection.call_args[1]
    assert call_args["collection_name"] == name
    assert isinstance(call_args["vectors_config"], VectorParams)
    assert call_args["vectors_config"].size == vector_size
    assert call_args["vectors_config"].distance == Distance.COSINE
```

### Example Unit Test for Utilities

```python
# tests/unit/test_utils.py
import pytest
import numpy as np
from mcp_server_qdrant.tools.utils import (
    combine_vectors, 
    calculate_cosine_similarity,
    text_to_chunks
)

@pytest.mark.unit
def test_combine_vectors():
    """Test the vector combination function."""
    # Arrange
    vectors = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ]
    weights = [0.7, 0.3]
    
    # Act
    result = combine_vectors(vectors, weights)
    
    # Assert
    expected = np.array([0.7, 0.3, 0.0])
    expected = expected / np.linalg.norm(expected)
    np.testing.assert_almost_equal(result, expected.tolist())

@pytest.mark.unit
def test_text_to_chunks():
    """Test the text chunking function."""
    # Arrange
    text = "This is a test. " * 100  # ~1400 characters
    chunk_size = 500
    chunk_overlap = 100
    
    # Act
    chunks = text_to_chunks(text, chunk_size, chunk_overlap)
    
    # Assert
    assert len(chunks) > 1
    assert len(chunks[0]) <= chunk_size
    
    # Check overlap
    if len(chunks) > 1:
        end_of_first = chunks[0][-chunk_overlap:]
        start_of_second = chunks[1][:chunk_overlap]
        assert end_of_first == start_of_second
```

## Integration Tests

Integration tests focus on the interaction between components and with Qdrant.

### Example Integration Test for Search

```python
# tests/integration/test_search_integration.py
import pytest
from mcp.server.fastmcp import Context
from mcp_server_qdrant.tools.search.nlq import natural_language_query
from mcp_server_qdrant.qdrant import Entry

@pytest.mark.integration
async def test_search_integration(test_qdrant_connector):
    """Test the search functionality with a real Qdrant instance."""
    # Arrange
    mock_ctx = Context()
    mock_ctx.request_context = type('obj', (object,), {
        'lifespan_context': {
            'qdrant_connector': test_qdrant_connector
        }
    })
    
    # Add some test data
    test_entries = [
        Entry(content="Python is a programming language", metadata={"type": "language"}),
        Entry(content="JavaScript is used for web development", metadata={"type": "language"}),
        Entry(content="Qdrant is a vector database", metadata={"type": "database"})
    ]
    
    for entry in test_entries:
        await test_qdrant_connector.store(entry)
    
    # Act
    results = await natural_language_query(
        ctx=mock_ctx,
        query="What programming languages are there?",
        limit=5
    )
    
    # Assert
    assert len(results) > 0
    # We expect the Python and JavaScript entries to be returned for this query
    languages_found = sum(1 for r in results if "Python" in r["document"] or "JavaScript" in r["document"])
    assert languages_found > 0
```

### Example Integration Test for Collection Management

```python
# tests/integration/test_collection_mgmt_integration.py
import pytest
from mcp.server.fastmcp import Context
from mcp_server_qdrant.tools.collection_mgmt.create_collection import create_collection
from mcp_server_qdrant.tools.collection_mgmt.migrate_collection import migrate_collection

@pytest.mark.integration
async def test_create_and_migrate_collection(test_qdrant_connector):
    """Test creating and migrating collections with a real Qdrant instance."""
    # Arrange
    mock_ctx = Context()
    mock_ctx.request_context = type('obj', (object,), {
        'lifespan_context': {
            'qdrant_connector': test_qdrant_connector
        }
    })
    
    # Act - Create source collection
    create_result = await create_collection(
        ctx=mock_ctx,
        name="source_collection",
        vector_size=384,
        distance="Cosine"
    )
    
    # Assert
    assert create_result["status"] == "success"
    
    # Add test data to source collection
    client = test_qdrant_connector._client
    embedding_provider = test_qdrant_connector._embedding_provider
    vector = await embedding_provider.embed_query("Test document")
    
    await client.upsert(
        collection_name="source_collection",
        points=[{
            "id": "1",
            "vector": vector,
            "payload": {"text": "Test document", "metadata": {"test": True}}
        }]
    )
    
    # Act - Create target collection and migrate
    await create_collection(
        ctx=mock_ctx,
        name="target_collection",
        vector_size=384,
        distance="Cosine"
    )
    
    migrate_result = await migrate_collection(
        ctx=mock_ctx,
        source_collection="source_collection",
        target_collection="target_collection"
    )
    
    # Assert
    assert migrate_result["status"] == "success"
    assert migrate_result["points_migrated"] == 1
    
    # Verify the point was migrated
    search_result = await client.search(
        collection_name="target_collection",
        query_vector=vector,
        limit=1
    )
    
    assert len(search_result) == 1
    assert search_result[0].payload["text"] == "Test document"
```

## End-to-End Tests

End-to-end tests verify that the complete MCP server functions as expected.

### Example E2E Test Setup

```python
# tests/e2e/conftest.py
import os
import pytest
import asyncio
import tempfile
import subprocess
import time
import json
import httpx
from mcp.client import MCPClient

@pytest.fixture(scope="module")
def mcp_server_process():
    """Start the MCP server process for testing."""
    # Create a temporary directory for Qdrant data
    temp_dir = tempfile.TemporaryDirectory()
    
    # Set up environment variables for the server
    env = os.environ.copy()
    env.update({
        "QDRANT_LOCAL_PATH": temp_dir.name,
        "COLLECTION_NAME": "test_collection",
        "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2"
    })
    
    # Start the server with SSE transport
    process = subprocess.Popen(
        ["python", "-m", "mcp_server_qdrant.server", "--transport", "sse"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for the server to start
    time.sleep(5)
    
    # Check if the server is running
    try:
        response = httpx.get("http://localhost:8000/health")
        assert response.status_code == 200
    except Exception as e:
        stdout, stderr = process.communicate(timeout=1)
        print(f"Failed to start MCP server: {e}")
        print(f"STDOUT: {stdout.decode()}")
        print(f"STDERR: {stderr.decode()}")
        process.terminate()
        raise
    
    yield process
    
    # Clean up
    process.terminate()
    process.wait(timeout=5)
    temp_dir.cleanup()

@pytest.fixture
async def mcp_client(mcp_server_process):
    """Create an MCP client connected to the test server."""
    client = MCPClient(transport_url="http://localhost:8000/sse")
    await client.connect()
    yield client
    await client.disconnect()
```

### Example E2E Test

```python
# tests/e2e/test_mcp_server.py
import pytest
import json

@pytest.mark.e2e
async def test_store_and_find(mcp_client):
    """Test storing and finding information via the MCP client."""
    # Act - Store information
    store_response = await mcp_client.execute_tool(
        tool_name="qdrant-store",
        params={"information": "E2E test information"}
    )
    
    # Assert
    assert "Remembered" in store_response
    
    # Act - Find information
    find_response = await mcp_client.execute_tool(
        tool_name="qdrant-find",
        params={"query": "test information"}
    )
    
    # Assert
    assert isinstance(find_response, list)
    assert len(find_response) > 1  # First item is the query result header
    assert "E2E test information" in " ".join(find_response)

@pytest.mark.e2e
async def test_nlq_search(mcp_client):
    """Test natural language query search via the MCP client."""
    # First store some test data
    await mcp_client.execute_tool(
        tool_name="qdrant-store",
        params={"information": "Artificial intelligence is revolutionizing industries"}
    )
    
    await mcp_client.execute_tool(
        tool_name="qdrant-store",
        params={"information": "Machine learning is a subset of AI"}
    )
    
    # Act - Search
    search_response = await mcp_client.execute_tool(
        tool_name="nlq-search",
        params={
            "query": "What is AI?",
            "limit": 5
        }
    )
    
    # Assert
    assert isinstance(search_response, list)
    assert len(search_response) > 0
    
    # At least one result should mention AI or artificial intelligence
    ai_mentioned = any(
        "artificial intelligence" in result["document"].lower() or
        "ai" in result["document"].lower()
        for result in search_response
    )
    assert ai_mentioned
```

## Performance Tests

Performance tests ensure that the MCP server meets performance requirements.

### Example Benchmark Test

```python
# tests/performance/test_search_performance.py
import pytest
import time
import statistics
from mcp_server_qdrant.tools.search.nlq import natural_language_query
from mcp_server_qdrant.qdrant import Entry

@pytest.mark.performance
async def test_search_performance(test_qdrant_connector):
    """Test the performance of search operations."""
    # Arrange
    mock_ctx = Context()
    mock_ctx.request_context = type('obj', (object,), {
        'lifespan_context': {
            'qdrant_connector': test_qdrant_connector
        }
    })
    
    # Add test data - 100 entries
    test_entries = [
        Entry(content=f"Test document {i}", metadata={"index": i})
        for i in range(100)
    ]
    
    for entry in test_entries:
        await test_qdrant_connector.store(entry)
    
    # Act - Measure search time over multiple iterations
    iterations = 10
    search_times = []
    
    for i in range(iterations):
        start_time = time.time()
        results = await natural_language_query(
            ctx=mock_ctx,
            query=f"Test document {i % 100}",
            limit=10
        )
        end_time = time.time()
        search_times.append((end_time - start_time) * 1000)  # Convert to ms
    
    # Assert
    assert len(results) > 0
    
    # Performance metrics
    avg_time = statistics.mean(search_times)
    p95_time = sorted(search_times)[int(iterations * 0.95)]
    
    # For comparison, not for failing the test
    print(f"Search performance:")
    print(f"  Average time: {avg_time:.2f} ms")
    print(f"  P95 time: {p95_time:.2f} ms")
    print(f"  Min time: {min(search_times):.2f} ms")
    print(f"  Max time: {max(search_times):.2f} ms")
    
    # Optional: Assert performance requirements
    # assert avg_time < 50.0, f"Average search time ({avg_time:.2f} ms) exceeds limit (50 ms)"
```

### Example Load Test

```python
# tests/performance/test_load.py
import pytest
import asyncio
import time
from mcp.client import MCPClient

@pytest.mark.performance
async def test_concurrent_requests(mcp_server_process):
    """Test the server's ability to handle concurrent requests."""
    # Arrange
    num_clients = 10
    clients = []
    
    # Create multiple clients
    for i in range(num_clients):
        client = MCPClient(transport_url="http://localhost:8000/sse")
        await client.connect()
        clients.append(client)
    
    try:
        # Act - Send requests concurrently
        start_time = time.time()
        tasks = []
        
        for i, client in enumerate(clients):
            task = asyncio.create_task(client.execute_tool(
                tool_name="qdrant-store",
                params={"information": f"Concurrent test {i}"}
            ))
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        # Assert
        total_time = (end_time - start_time) * 1000  # Convert to ms
        
        # All requests should succeed
        assert all("Remembered" in result for result in results)
        
        # Check if time is reasonable (adjust based on your performance expectations)
        avg_time_per_request = total_time / num_clients
        print(f"Concurrent performance ({num_clients} clients):")
        print(f"  Total time: {total_time:.2f} ms")
        print(f"  Average time per request: {avg_time_per_request:.2f} ms")
        
        # Optional: Assert performance requirements
        # assert avg_time_per_request < 100.0, f"Average request time ({avg_time_per_request:.2f} ms) exceeds limit (100 ms)"
    
    finally:
        # Clean up
        for client in clients:
            await client.disconnect()
```

## Test Automation and CI/CD

Automate testing with GitHub Actions to run tests on every push and pull request.

### Example GitHub Actions Workflow

```yaml
# .github/workflows/test.yml
name: Test

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[test]"
        
    - name: Run unit tests
      run: |
        pytest tests/unit -v
        
    - name: Run integration tests
      run: |
        pytest tests/integration -v
        
    - name: Run E2E tests
      run: |
        pytest tests/e2e -v
        
    - name: Generate coverage report
      run: |
        pytest --cov=src/mcp_server_qdrant --cov-report=xml
        
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

## Manual Testing Guide

For features that are difficult to automate, a manual testing process is useful.

### MCP Inspector Testing

Use the MCP Inspector for interactive testing:

1. Start the server:
   ```bash
   QDRANT_URL=":memory:" COLLECTION_NAME="test" \
   mcp dev src/mcp_server_qdrant/server.py
   ```

2. Open the inspector in your browser at http://localhost:5173

3. Test each tool with the following workflow:
   - Send a request with appropriate parameters
   - Verify the response is as expected
   - Check server logs for any errors or warnings

### Test Scenarios for Each Tool Category

#### Search Tools

1. **Natural Language Query**
   - Test with simple queries
   - Test with complex queries
   - Test with filters and different collections

2. **Hybrid Search**
   - Test with queries that benefit from keyword matching
   - Test with different text field names
   - Verify results are properly re-ranked

#### Collection Management Tools

1. **Create Collection**
   - Create collections with different parameters
   - Verify collections are created correctly
   - Test error handling for duplicate collections

2. **Migrate Collection**
   - Migrate between collections
   - Test with transformation functions
   - Verify all data is migrated correctly

#### Advanced Query Tools

1. **Semantic Router**
   - Test with queries matching different routes
   - Test threshold behavior
   - Verify confidence scores are reasonable

2. **Query Decomposition**
   - Test simple and complex queries
   - Test rule-based and LLM-based decomposition
   - Verify subqueries are meaningful

## Test Data Generation

Generate test data for comprehensive testing:

```python
# tests/utils/generate_test_data.py
import random
import json
from typing import List, Dict, Any

def generate_test_documents(count: int = 100) -> List[Dict[str, Any]]:
    """Generate test documents with diverse content."""
    topics = ["technology", "science", "history", "art", "business"]
    documents = []
    
    for i in range(count):
        topic = random.choice(topics)
        
        if topic == "technology":
            content = f"Document {i}: Discussing advancements in {random.choice(['AI', 'blockchain', 'cloud computing', 'IoT'])}"
        elif topic == "science":
            content = f"Document {i}: Research about {random.choice(['quantum physics', 'biology', 'chemistry', 'astronomy'])}"
        elif topic == "history":
            content = f"Document {i}: Historical events from the {random.choice(['ancient', 'medieval', 'renaissance', 'modern'])} era"
        elif topic == "art":
            content = f"Document {i}: Exploring {random.choice(['painting', 'sculpture', 'music', 'literature'])} techniques"
        else:  # business
            content = f"Document {i}: Analysis of {random.choice(['market trends', 'finance', 'management', 'startups'])}"
        
        documents.append({
            "id": str(i),
            "content": content,
            "metadata": {
                "topic": topic,
                "length": len(content),
                "priority": random.randint(1, 5)
            }
        })
    
    return documents

def save_test_data(documents: List[Dict[str, Any]], filename: str = "test_data.json") -> None:
    """Save generated test data to a file."""
    with open(filename, "w") as f:
        json.dump(documents, f, indent=2)

if __name__ == "__main__":
    documents = generate_test_documents(200)
    save_test_data(documents)
    print(f"Generated {len(documents)} test documents")
```

## Common Testing Issues

### Asynchronous Testing

- Use `pytest-asyncio` properly
- Ensure all async functions are properly awaited
- Handle connection cleanup in fixtures

```python
# Example of proper async test
@pytest.mark.asyncio
async def test_async_function():
    # Arrange
    ...
    
    # Act
    result = await my_async_function()
    
    # Assert
    assert result == expected
```

### Mocking Challenges

- Mock the correct level of abstraction
- Ensure mock return values match expected types
- Properly mock async functions

```python
# Example of mocking async functions
async def test_with_async_mock():
    mock_func = AsyncMock()
    mock_func.return_value = expected_result
    
    # Use patch for functions imported in the module under test
    with patch('module.async_function', mock_func):
        result = await function_that_calls_async_function()
    
    mock_func.assert_called_once()
    assert result == expected_transformed_result
```

### Integration Test State Management

- Ensure tests are isolated
- Use fixtures with proper scope
- Clean up all resources after tests

```python
# Example of proper resource cleanup
@pytest.fixture
async def resource_fixture():
    # Setup resource
    resource = await create_resource()
    
    yield resource
    
    # Cleanup
    try:
        await cleanup_resource(resource)
    except Exception as e:
        print(f"Error during cleanup: {e}")
```

By implementing this comprehensive testing strategy, you'll ensure the reliability and performance of your enhanced Qdrant MCP server, making it robust for production use.
