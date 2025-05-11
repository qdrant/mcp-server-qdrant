# Adding OpenAI Embedding Support to mcp-server-qdrant

This guide provides a detailed approach for extending the mcp-server-qdrant project to support OpenAI's embedding models, including the `text-embedding-3-small` model. The current implementation only supports FastEmbed models, so we'll need to create a new embedding provider implementation.

## Table of Contents

1. [Overview](#overview)
2. [Implementation Steps](#implementation-steps)
   - [Create OpenAI Embedding Provider](#1-create-openai-embedding-provider)
   - [Update Embedding Provider Type Enum](#2-update-embedding-provider-type-enum)
   - [Modify Provider Factory](#3-modify-provider-factory)
   - [Update Settings Class](#4-update-settings-class)
   - [Add Required Dependencies](#5-add-required-dependencies)
   - [Update Dockerfile](#6-update-dockerfile)
3. [Testing](#testing)
4. [Usage](#usage)
   - [Command Line](#command-line)
   - [Docker](#docker)
   - [Claude Desktop Configuration](#claude-desktop-configuration)
   - [VS Code Configuration](#vs-code-configuration)
5. [Advanced Considerations](#advanced-considerations)
   - [Error Handling and Retries](#error-handling-and-retries)
   - [Caching](#caching)
   - [Batching](#batching)
   - [Cost Considerations](#cost-considerations)
6. [Documentation Updates](#documentation-updates)

## Overview

The mcp-server-qdrant project follows a modular architecture that makes it relatively straightforward to add support for new embedding models. The project has a clear extension point for new embedding providers through its abstract `EmbeddingProvider` base class.

Our approach will:

1. Create a new embedding provider implementation for OpenAI's models
2. Register the new provider in the factory
3. Update settings to handle OpenAI API keys
4. Update the environment variable handling
5. Add required dependencies

## Implementation Steps

### 1. Create OpenAI Embedding Provider

First, create a new file `src/mcp_server_qdrant/embeddings/openai.py` with the following content:

```python
import asyncio
from typing import List
import httpx

from mcp_server_qdrant.embeddings.base import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI implementation of the embedding provider.
    :param model_name: The name of the OpenAI embedding model to use (e.g., 'text-embedding-3-small').
    :param api_key: The OpenAI API key to use for authentication.
    """

    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key
        # Mapping of model names to their dimensions
        self.dimension_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        if model_name not in self.dimension_map:
            raise ValueError(f"Unsupported model: {model_name}. Supported models: {', '.join(self.dimension_map.keys())}")

    async def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed a list of documents into vectors."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={"input": documents, "model": self.model_name},
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
            # Sort by index to ensure the order matches the input
            embeddings_data = sorted(data["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in embeddings_data]

    async def embed_query(self, query: str) -> List[float]:
        """Embed a query into a vector."""
        embeddings = await self.embed_documents([query])
        return embeddings[0]

    def get_vector_name(self) -> str:
        """
        Return the name of the vector for the Qdrant collection.
        """
        return f"openai-{self.model_name.replace('/', '-')}"

    def get_vector_size(self) -> int:
        """Get the size of the vector for the Qdrant collection."""
        return self.dimension_map[self.model_name]
```

This implementation:
- Handles the OpenAI API authentication
- Implements all required methods from the base `EmbeddingProvider` class
- Maps model names to their respective embedding dimensions
- Uses httpx for asynchronous HTTP requests

### 2. Update Embedding Provider Type Enum

Update the `src/mcp_server_qdrant/embeddings/types.py` file to include the new provider type:

```python
from enum import Enum


class EmbeddingProviderType(Enum):
    FASTEMBED = "fastembed"
    OPENAI = "openai"
```

This adds the OpenAI provider type to the enum used throughout the application.

### 3. Modify Provider Factory

Update the factory in `src/mcp_server_qdrant/embeddings/factory.py` to handle the new provider:

```python
from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.embeddings.types import EmbeddingProviderType
from mcp_server_qdrant.settings import EmbeddingProviderSettings


def create_embedding_provider(settings: EmbeddingProviderSettings) -> EmbeddingProvider:
    """
    Create an embedding provider based on the specified type.
    :param settings: The settings for the embedding provider.
    :return: An instance of the specified embedding provider.
    """
    if settings.provider_type == EmbeddingProviderType.FASTEMBED:
        from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider

        return FastEmbedProvider(settings.model_name)
    elif settings.provider_type == EmbeddingProviderType.OPENAI:
        from mcp_server_qdrant.embeddings.openai import OpenAIEmbeddingProvider

        if not settings.api_key:
            raise ValueError("API key is required for OpenAI embedding provider. Set the OPENAI_API_KEY environment variable.")
        return OpenAIEmbeddingProvider(settings.model_name, settings.api_key)
    else:
        raise ValueError(f"Unsupported embedding provider: {settings.provider_type}. Supported providers: {[t.value for t in EmbeddingProviderType]}")
```

This updated factory:
- Imports the new OpenAI provider when needed
- Validates the API key is present
- Returns the appropriate provider instance

### 4. Update Settings Class

Modify the `src/mcp_server_qdrant/settings.py` file to include the OpenAI API key:

```python
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings

from mcp_server_qdrant.embeddings.types import EmbeddingProviderType

# ... existing code ...

class EmbeddingProviderSettings(BaseSettings):
    """
    Configuration for the embedding provider.
    """

    provider_type: EmbeddingProviderType = Field(
        default=EmbeddingProviderType.FASTEMBED,
        validation_alias="EMBEDDING_PROVIDER",
    )
    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        validation_alias="EMBEDDING_MODEL",
    )
    api_key: Optional[str] = Field(
        default=None,
        validation_alias="OPENAI_API_KEY",
    )
```

This adds the OpenAI API key to the settings, using the environment variable `OPENAI_API_KEY`.

### 5. Add Required Dependencies

Update the `pyproject.toml` file to include the httpx dependency:

```toml
[project]
name = "mcp-server-qdrant"
version = "0.7.1"
description = "MCP server for retrieving context from a Qdrant vector database"
readme = "README.md"
requires-python = ">=3.10"
license = "Apache-2.0"
dependencies = [
    "mcp[cli]>=1.3.0",
    "fastembed>=0.6.0",
    "qdrant-client>=1.12.0",
    "pydantic>=2.10.6",
    "httpx>=0.28.0",  # Add this line
]

# ... rest of the file ...
```

This ensures that the httpx library is installed for making HTTP requests to the OpenAI API.

### 6. Update Dockerfile

Update the Dockerfile to include the new environment variable and dependency:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install uv for package management
RUN pip install --no-cache-dir uv

# Install the mcp-server-qdrant package and httpx dependency
RUN uv pip install --system --no-cache-dir mcp-server-qdrant httpx

# Expose the default port for SSE transport
EXPOSE 8000

# Set environment variables with defaults that can be overridden at runtime
ENV QDRANT_URL=""
ENV QDRANT_API_KEY=""
ENV COLLECTION_NAME="default-collection"
ENV EMBEDDING_PROVIDER="fastembed"
ENV EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
ENV OPENAI_API_KEY=""

# Run the server with SSE transport
CMD uvx mcp-server-qdrant --transport sse
```

This adds the `OPENAI_API_KEY` environment variable with a default empty value and ensures httpx is installed.

## Testing

To test the OpenAI embedding provider, create a new test file `tests/test_openai_provider.py`:

```python
import os
import pytest
from unittest.mock import patch, MagicMock

from mcp_server_qdrant.embeddings.openai import OpenAIEmbeddingProvider


@pytest.mark.asyncio
class TestOpenAIProviderUnit:
    """Unit tests for OpenAIEmbeddingProvider."""

    def test_initialization(self):
        """Test that the provider can be initialized with a valid model."""
        provider = OpenAIEmbeddingProvider("text-embedding-3-small", "test-api-key")
        assert provider.model_name == "text-embedding-3-small"
        assert provider.api_key == "test-api-key"

    def test_invalid_model(self):
        """Test that initialization with an invalid model raises an error."""
        with pytest.raises(ValueError, match="Unsupported model"):
            OpenAIEmbeddingProvider("invalid-model", "test-api-key")

    def test_get_vector_name(self):
        """Test that the vector name is generated correctly."""
        provider = OpenAIEmbeddingProvider("text-embedding-3-small", "test-api-key")
        vector_name = provider.get_vector_name()
        assert vector_name == "openai-text-embedding-3-small"

    def test_get_vector_size(self):
        """Test that the vector size is correct for different models."""
        provider1 = OpenAIEmbeddingProvider("text-embedding-3-small", "test-api-key")
        assert provider1.get_vector_size() == 1536

        provider2 = OpenAIEmbeddingProvider("text-embedding-3-large", "test-api-key")
        assert provider2.get_vector_size() == 3072

        provider3 = OpenAIEmbeddingProvider("text-embedding-ada-002", "test-api-key")
        assert provider3.get_vector_size() == 1536

    @patch("httpx.AsyncClient.post")
    async def test_embed_documents(self, mock_post):
        """Test that documents can be embedded."""
        # Mock response
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"index": 0, "embedding": [0.1, 0.2, 0.3]},
                {"index": 1, "embedding": [0.4, 0.5, 0.6]},
            ]
        }
        mock_post.return_value = mock_response

        provider = OpenAIEmbeddingProvider("text-embedding-3-small", "test-api-key")
        documents = ["Test document 1", "Test document 2"]
        embeddings = await provider.embed_documents(documents)

        # Check the API call
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer test-api-key"
        assert kwargs["json"]["model"] == "text-embedding-3-small"
        assert kwargs["json"]["input"] == documents

        # Check the result
        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]

    @patch("httpx.AsyncClient.post")
    async def test_embed_query(self, mock_post):
        """Test that a query can be embedded."""
        # Mock response
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"index": 0, "embedding": [0.1, 0.2, 0.3]},
            ]
        }
        mock_post.return_value = mock_response

        provider = OpenAIEmbeddingProvider("text-embedding-3-small", "test-api-key")
        query = "Test query"
        embedding = await provider.embed_query(query)

        # Check the API call
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert kwargs["json"]["input"] == [query]

        # Check the result
        assert embedding == [0.1, 0.2, 0.3]
```

These tests verify the basic functionality of the OpenAI embedding provider without making actual API calls.

## Usage

After implementing the changes, you can use the OpenAI embedding model in several ways:

### Command Line

Run the server with the OpenAI embedding model:

```bash
QDRANT_URL="http://localhost:6333" \
COLLECTION_NAME="my-collection" \
EMBEDDING_PROVIDER="openai" \
EMBEDDING_MODEL="text-embedding-3-small" \
OPENAI_API_KEY="your-openai-api-key" \
uvx mcp-server-qdrant --transport sse
```

### Docker

Run the server using Docker:

```bash
docker run -p 8000:8000 \
  -e QDRANT_URL="http://your-qdrant-server:6333" \
  -e QDRANT_API_KEY="your-qdrant-api-key" \
  -e COLLECTION_NAME="your-collection" \
  -e EMBEDDING_PROVIDER="openai" \
  -e EMBEDDING_MODEL="text-embedding-3-small" \
  -e OPENAI_API_KEY="your-openai-api-key" \
  mcp-server-qdrant
```

### Claude Desktop Configuration

To use the OpenAI embedding model with Claude Desktop, update the `claude_desktop_config.json` file:

```json
{
  "qdrant": {
    "command": "uvx",
    "args": ["mcp-server-qdrant"],
    "env": {
      "QDRANT_URL": "https://xyz-example.eu-central.aws.cloud.qdrant.io:6333",
      "QDRANT_API_KEY": "your_qdrant_api_key",
      "COLLECTION_NAME": "your-collection-name",
      "EMBEDDING_PROVIDER": "openai",
      "EMBEDDING_MODEL": "text-embedding-3-small",
      "OPENAI_API_KEY": "your_openai_api_key"
    }
  }
}
```

### VS Code Configuration

For VS Code, add the following configuration to the User Settings (JSON) file:

```json
{
  "mcp": {
    "inputs": [
      {
        "type": "promptString",
        "id": "qdrantUrl",
        "description": "Qdrant URL"
      },
      {
        "type": "promptString",
        "id": "qdrantApiKey",
        "description": "Qdrant API Key",
        "password": true
      },
      {
        "type": "promptString",
        "id": "collectionName",
        "description": "Collection Name"
      },
      {
        "type": "promptString",
        "id": "openaiApiKey",
        "description": "OpenAI API Key",
        "password": true
      }
    ],
    "servers": {
      "qdrant": {
        "command": "uvx",
        "args": ["mcp-server-qdrant"],
        "env": {
          "QDRANT_URL": "${input:qdrantUrl}",
          "QDRANT_API_KEY": "${input:qdrantApiKey}",
          "COLLECTION_NAME": "${input:collectionName}",
          "EMBEDDING_PROVIDER": "openai",
          "EMBEDDING_MODEL": "text-embedding-3-small",
          "OPENAI_API_KEY": "${input:openaiApiKey}"
        }
      }
    }
  }
}
```

## Advanced Considerations

### Error Handling and Retries

For a more robust implementation, consider adding error handling and retries to the OpenAI provider:

```python
import asyncio
from typing import List
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from mcp_server_qdrant.embeddings.base import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    # ... existing code ...

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError))
    )
    async def _make_embedding_request(self, documents: List[str]) -> List[List[float]]:
        """Make a request to the OpenAI API with retries."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={"input": documents, "model": self.model_name},
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
            # Sort by index to ensure the order matches the input
            embeddings_data = sorted(data["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in embeddings_data]

    async def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed a list of documents into vectors with retries."""
        return await self._make_embedding_request(documents)

    async def embed_query(self, query: str) -> List[float]:
        """Embed a query into a vector with retries."""
        embeddings = await self._make_embedding_request([query])
        return embeddings[0]
```

To use this implementation, add the tenacity library to your dependencies.

### Caching

To reduce API calls and improve performance, consider adding caching:

```python
import hashlib
import json
from functools import lru_cache
from typing import List, Dict, Any

class OpenAIEmbeddingProvider(EmbeddingProvider):
    # ... existing code ...

    def __init__(self, model_name: str, api_key: str, cache_size: int = 1000):
        # ... existing initialization ...
        self._cache = lru_cache(maxsize=cache_size)(self._get_embedding)

    def _compute_hash(self, text: str) -> str:
        """Compute a hash for the text to use as a cache key."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    async def _get_embedding(self, text_hash: str, text: str) -> List[float]:
        """Get the embedding for a single text, bypass cache."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={"input": [text], "model": self.model_name},
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
            return data["data"][0]["embedding"]

    async def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed a list of documents into vectors, using cache when possible."""
        results = []
        for doc in documents:
            doc_hash = self._compute_hash(doc)
            embedding = await self._cache(doc_hash, doc)
            results.append(embedding)
        return results

    async def embed_query(self, query: str) -> List[float]:
        """Embed a query into a vector, using cache when possible."""
        query_hash = self._compute_hash(query)
        return await self._cache(query_hash, query)
```

### Batching

OpenAI has limits on the size of requests. For large document sets, implement batching:

```python
import asyncio
from typing import List

class OpenAIEmbeddingProvider(EmbeddingProvider):
    # ... existing code ...

    async def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed a list of documents into vectors with batching."""
        batch_size = 20  # Adjust based on token limits
        results = []
        
        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={"input": batch, "model": self.model_name},
                    timeout=30.0,
                )
                response.raise_for_status()
                data = response.json()
                
                # Sort by index to ensure the order matches the input
                embeddings_data = sorted(data["data"], key=lambda x: x["index"])
                batch_results = [item["embedding"] for item in embeddings_data]
                results.extend(batch_results)
                
                # Avoid rate limits
                if i + batch_size < len(documents):
                    await asyncio.sleep(0.5)
                    
        return results
```

### Cost Considerations

OpenAI's embedding API is not free. Consider adding cost tracking and limits:

```python
import time
from typing import Dict, List

class OpenAIEmbeddingProvider(EmbeddingProvider):
    # ... existing code ...

    def __init__(self, model_name: str, api_key: str):
        # ... existing initialization ...
        self.token_count = 0
        self.request_count = 0
        self.last_reset = time.time()
        
        # Token counting estimates - adjust based on actual values
        self.token_estimators = {
            "text-embedding-3-small": lambda text: len(text.split()) * 1.3,
            "text-embedding-3-large": lambda text: len(text.split()) * 1.3,
            "text-embedding-ada-002": lambda text: len(text.split()) * 1.3,
        }
        
    def _estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in a text."""
        estimator = self.token_estimators.get(self.model_name)
        if estimator:
            return int(estimator(text))
        return len(text.split())  # Fallback

    async def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed documents and track usage."""
        # Track token usage
        for doc in documents:
            self.token_count += self._estimate_tokens(doc)
        
        self.request_count += 1
        
        # Reset counters daily
        if time.time() - self.last_reset > 86400:  # 24 hours
            self.token_count = 0
            self.request_count = 0
            self.last_reset = time.time()
            
        # Proceed with embedding
        # ... existing embedding code ...
```

## Documentation Updates

Finally, update the README.md file to document the new OpenAI provider support:

```markdown
## OpenAI Embedding Models

As of version X.Y.Z, the server now supports OpenAI embedding models through the OpenAI API. You can use models such as `text-embedding-3-small`, `text-embedding-3-large`, and `text-embedding-ada-002`.

### Configuration

To use OpenAI embedding models, set the following environment variables:

| Name                 | Description                                         | Example Value                 |
|----------------------|-----------------------------------------------------|-------------------------------|
| `EMBEDDING_PROVIDER` | The embedding provider to use                       | `openai`                      |
| `EMBEDDING_MODEL`    | The OpenAI model to use for embeddings              | `text-embedding-3-small`      |
| `OPENAI_API_KEY`     | Your OpenAI API key                                 | `sk-...`                      |

### Example Usage

```bash
QDRANT_URL="http://localhost:6333" \
COLLECTION_NAME="my-collection" \
EMBEDDING_PROVIDER="openai" \
EMBEDDING_MODEL="text-embedding-3-small" \
OPENAI_API_KEY="your-openai-api-key" \
uvx mcp-server-qdrant
```

### Supported Models

| Model Name             | Dimensions | Best For                                      |
|------------------------|------------|-----------------------------------------------|
| text-embedding-3-small | 1536       | Most use cases                                |
| text-embedding-3-large | 3072       | Highest accuracy but more expensive           |
| text-embedding-ada-002 | 1536       | Legacy model (consider using newer ones)      |

### Cost Considerations

Unlike the built-in FastEmbed models, OpenAI models require API calls that have associated costs. Please refer to [OpenAI's pricing page](https://openai.com/pricing) for the current rates.
```

This completes the implementation of OpenAI's embedding model support for the mcp-server-qdrant project.
