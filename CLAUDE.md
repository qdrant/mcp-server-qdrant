# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

mcp-server-qdrant is a server that retrieves context from a Qdrant vector database for various Anthropic AI clients, including Claude Desktop and VS Code. It supports embedding models from different providers to convert text into vector embeddings.

## Architecture

- **Core Components**:
  - `EmbeddingProvider`: Abstract base class that all embedding providers extend
  - `FastEmbedProvider`: Default implementation using FastEmbed models
  - `OpenAIEmbeddingProvider`: Implementation for OpenAI embedding models
  - Provider factory pattern for creating appropriate embedding providers
  - Qdrant integration for vector storage and retrieval

- **Key Modules**:
  - `embeddings/`: Contains embedding provider implementations
  - `settings.py`: Configuration using Pydantic BaseSettings for environment variables

## Environment Configuration

The server uses these environment variables:
- `QDRANT_URL`: URL of the Qdrant server
- `QDRANT_API_KEY`: API key for Qdrant (if required)
- `COLLECTION_NAME`: Name of the Qdrant collection (default: "default-collection")
- `EMBEDDING_PROVIDER`: Provider type ("fastembed" or "openai")
- `EMBEDDING_MODEL`: Model name (e.g., "sentence-transformers/all-MiniLM-L6-v2" or "text-embedding-3-small")
- `OPENAI_API_KEY`: OpenAI API key (required when using OpenAI provider)

## Development Commands

- Install dependencies:
  ```bash
  pip install -e ".[dev]"
  # Or using uv
  uv pip install -e ".[dev]"
  ```

- Run tests:
  ```bash
  pytest
  ```

- Run a specific test:
  ```bash
  pytest tests/test_specific_file.py::TestClass::test_specific_function
  ```

- Run the server:
  ```bash
  uvx mcp-server-qdrant --transport sse
  ```

- Run with specific configuration:
  ```bash
  QDRANT_URL="http://localhost:6333" \
  COLLECTION_NAME="my-collection" \
  EMBEDDING_PROVIDER="openai" \
  EMBEDDING_MODEL="text-embedding-3-small" \
  OPENAI_API_KEY="your-openai-api-key" \
  uvx mcp-server-qdrant --transport sse
  ```

## Extending the Codebase

When implementing new embedding providers:
1. Create a new provider class that extends `EmbeddingProvider`
2. Implement all required methods: `embed_documents`, `embed_query`, `get_vector_name`, `get_vector_size`
3. Add the provider type to the `EmbeddingProviderType` enum
4. Update the provider factory to handle the new type
5. Update settings to include any new configuration parameters
6. Add appropriate tests for the new provider