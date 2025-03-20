# mcp-server-qdrant: A Qdrant MCP server
[![smithery badge](https://smithery.ai/badge/mcp-server-qdrant)](https://smithery.ai/protocol/mcp-server-qdrant)

> The [Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction) is an open protocol that enables
> seamless integration between LLM applications and external data sources and tools. Whether you're building an
> AI-powered IDE, enhancing a chat interface, or creating custom AI workflows, MCP provides a standardized way to
> connect LLMs with the context they need.

This repository is an example of how to create a MCP server for [Qdrant](https://qdrant.tech/), a vector search engine.

<a href="https://glama.ai/mcp/servers/9ejy5scw5i"><img width="380" height="200" src="https://glama.ai/mcp/servers/9ejy5scw5i/badge" alt="mcp-server-qdrant MCP server" /></a>

## Overview

A powerful Model Context Protocol server for managing vector search operations with the Qdrant vector database.
It provides a comprehensive set of tools for similarity search, collection management, data processing, advanced queries, analytics, and document processing.

## Components

### Core Tools

1. `qdrant-store-memory`
   - Store a memory in the Qdrant database
   - Input:
     - `information` (string): Memory to store
   - Returns: Confirmation message
2. `qdrant-find-memories`
   - Retrieve a memory from the Qdrant database
   - Input:
     - `query` (string): Query to use for searching
   - Returns: Information stored in the Qdrant database as separate messages

### Search Tools

3. `nlq-search`
   - Search Qdrant using natural language
   - Input:
     - `query` (string): The natural language query
     - `collection` (string, optional): The collection to search
     - `limit` (int, default=10): Maximum number of results
     - `filter` (JSON, optional): Additional filters
     - `with_payload` (boolean or string array, default=true): Whether to include payload
   - Returns: Matched documents with scores and payloads

4. `hybrid-search`
   - Perform hybrid vector and keyword search for better results
   - Input:
     - `query` (string): The search query
     - `collection` (string): The collection to search
     - `textFieldName` (string): Field to use for keyword search
     - `limit` (int, optional): Maximum results to return
     - `filter` (JSON, optional): Additional filters
   - Returns: Combined results from vector and keyword search

5. `multi-vector-search`
   - Search using multiple query vectors with weights
   - Input:
     - `queries` (string[]): List of queries to combine
     - `collection` (string): The collection to search
     - `weights` (number[], optional): Weight for each query
     - `limit` (int, optional): Maximum results to return
   - Returns: Search results using the combined vector

### Collection Management Tools

6. `create-collection`
   - Create a new vector collection with specified parameters
   - Input:
     - `name` (string): Name for the new collection
     - `vector_size` (int): Size of vector embeddings
     - `distance` (string, optional): Distance metric ("Cosine", "Euclid", "Dot")
     - `hnsw_config` (JSON, optional): HNSW index configuration
     - `optimizers_config` (JSON, optional): Optimizers configuration
   - Returns: Status of the collection creation

7. `migrate-collection`
   - Migrate data between collections with optional transformations
   - Input:
     - `source_collection` (string): Source collection name
     - `target_collection` (string): Target collection name
     - `batch_size` (int, optional): Number of points per batch
     - `transform_fn` (string, optional): Python code for point transformation
   - Returns: Migration status and statistics

### Data Processing Tools

8. `batch-embed`
   - Generate embeddings for multiple texts in batch
   - Input:
     - `texts` (string[]): List of texts to embed
     - `model` (string, optional): Model to use for embedding
   - Returns: Generated embeddings

9. `chunk-and-process`
   - Split text into chunks, generate embeddings, and store in Qdrant
   - Input:
     - `text` (string): Text to chunk and process
     - `chunk_size` (int, optional): Size of each chunk
     - `chunk_overlap` (int, optional): Overlap between chunks
     - `collection` (string, optional): Collection to store in
     - `metadata` (JSON, optional): Additional metadata
   - Returns: Processed chunks with embeddings

### Advanced Query Tools

10. `semantic-router`
    - Route a query to the most semantically similar destination
    - Input:
      - `query` (string): The query to route
      - `routes` (array): List of routes with name and description
      - `threshold` (number, optional): Minimum similarity threshold
    - Returns: Best matching route and confidence score

11. `decompose-query`
    - Decompose a complex query into multiple simpler subqueries
    - Input:
      - `query` (string): Complex query to decompose
      - `use_llm` (boolean, optional): Whether to use LLM for decomposition
    - Returns: List of simpler subqueries

### Analytics Tools

12. `analyze-collection`
    - Analyze a Qdrant collection to extract statistics and schema information
    - Input:
      - `collection` (string): Collection to analyze
      - `sample_size` (int, optional): Number of points to sample
    - Returns: Collection statistics and schema information

13. `benchmark-query`
    - Benchmark query performance by running multiple iterations
    - Input:
      - `query` (string): Query to benchmark
      - `collection` (string): Collection to query
      - `iterations` (int, optional): Number of iterations
    - Returns: Performance metrics and statistics

### Document Processing Tools

14. `index-document`
    - Fetch, process, and index a document from a URL
    - Input:
      - `document_url` (string): URL of document to index
      - `collection` (string): Collection to store in
      - `chunk_size` (int, optional): Size of each chunk
      - `chunk_overlap` (int, optional): Overlap between chunks
      - `metadata` (JSON, optional): Additional metadata
    - Returns: Indexing results and statistics

15. `process-pdf`
    - Process a PDF file, extract content, and optionally store in Qdrant
    - Input:
      - `pdf_path` (string): Path to the PDF file
      - `collection` (string, optional): Collection to store in
      - `extract_tables` (boolean, optional): Whether to extract tables
      - `extract_images` (boolean, optional): Whether to extract images
    - Returns: Processing results and document information

## Environment Variables

The configuration of the server is done using environment variables:

| Name                     | Description                                                         | Default Value                                                     |
|--------------------------|---------------------------------------------------------------------|-------------------------------------------------------------------|
| `QDRANT_URL`             | URL of the Qdrant server                                            | None                                                              |
| `QDRANT_API_KEY`         | API key for the Qdrant server                                       | None                                                              |
| `COLLECTION_NAME`        | Name of the collection to use                                       | *Required*                                                        |
| `QDRANT_LOCAL_PATH`      | Path to the local Qdrant database (alternative to `QDRANT_URL`)     | None                                                              |
| `EMBEDDING_PROVIDER`     | Embedding provider to use (currently only "fastembed" is supported) | `fastembed`                                                       |
| `EMBEDDING_MODEL`        | Name of the embedding model to use                                  | `sentence-transformers/all-MiniLM-L6-v2`                          |
| `DEFAULT_VECTOR_SIZE`    | Default vector size for new collections                             | 384                                                               |
| `DEFAULT_DISTANCE`       | Default distance metric for new collections                         | "Cosine"                                                          |
| `PDF_EXTRACT_TABLES`     | Whether to extract tables from PDFs by default                      | false                                                             |
| `PDF_EXTRACT_IMAGES`     | Whether to extract images from PDFs by default                      | false                                                             |
| `TOOL_STORE_DESCRIPTION` | Custom description for the store tool                               | See default in [`settings.py`](src/mcp_server_qdrant/settings.py) |
| `TOOL_FIND_DESCRIPTION`  | Custom description for the find tool                                | See default in [`settings.py`](src/mcp_server_qdrant/settings.py) |

You can also customize descriptions for all the other tools by setting environment variables like `NLQ_SEARCH_DESCRIPTION`, `HYBRID_SEARCH_DESCRIPTION`, etc.

Note: You cannot provide both `QDRANT_URL` and `QDRANT_LOCAL_PATH` at the same time.

> [!IMPORTANT]
> Command-line arguments are not supported anymore! Please use environment variables for all configuration.

## Installation

### Using uv (recommended)

When using [`uv`](https://docs.astral.sh/uv/) no specific installation is needed to directly run *mcp-server-qdrant*.

```shell
uv run mcp-server-qdrant \
  --qdrant-url "http://localhost:6333" \
  --qdrant-api-key "your_api_key" \
  --collection-name "my_collection" \
  --embedding-model "sentence-transformers/all-MiniLM-L6-v2"
```

### Installing via Smithery

To install Qdrant MCP Server for Claude Desktop automatically via [Smithery](https://smithery.ai/protocol/mcp-server-qdrant):

```bash
npx @smithery/cli install mcp-server-qdrant --client claude
```

## Usage with Claude Desktop

To use this server with the Claude Desktop app, add the following configuration to the "mcpServers" section of your `claude_desktop_config.json`:

```json
{
  "qdrant": {
    "command": "uvx",
    "args": [
      "mcp-server-qdrant",
      "--qdrant-url",
      "http://localhost:6333",
      "--qdrant-api-key",
      "your_api_key",
      "--collection-name",
      "your_collection_name"
    ]
  }
}
```

Replace `http://localhost:6333`, `your_api_key` and `your_collection_name` with your Qdrant server URL, Qdrant API key
and collection name, respectively. The use of API key is optional, but recommended for security reasons, and depends on
the Qdrant server configuration.

This MCP server will automatically create a collection with the specified name if it doesn't exist.

By default, the server will use the `sentence-transformers/all-MiniLM-L6-v2` embedding model to encode memories.
For the time being, only [FastEmbed](https://qdrant.github.io/fastembed/) models are supported, and you can change it
by passing the `--embedding-model` argument to the server.

## Use Cases

### Document Management and Retrieval

Use the document processing tools to index and search through your documents:

```bash
# Indexing documents for future search
QDRANT_URL="http://localhost:6333" \
COLLECTION_NAME="documents" \
uvx mcp-server-qdrant
```

The MCP server will help with:
- Processing and chunking documents from various sources
- Extracting text, tables, and images from PDFs
- Performing semantic searches across your document collection
- Finding relevant information based on natural language queries

### Vector Database Management

Use the collection and analytics tools to manage your vector database:

```bash
# Vector database management setup
QDRANT_URL="http://localhost:6333" \
COLLECTION_NAME="vectors" \
uvx mcp-server-qdrant
```

This setup provides:
- Collection creation and migration capabilities
- Performance benchmarking for optimizing queries
- Schema and vector statistics for collection analysis
- Data processing tools for working with vectors

### Knowledge Base and FAQ System

Build a knowledge base system using the semantic routing and hybrid search capabilities:

```bash
# Knowledge base setup
QDRANT_URL="http://localhost:6333" \
COLLECTION_NAME="knowledge-base" \
uvx mcp-server-qdrant
```

Features for this use case:
- Store facts, articles, and information with metadata
- Use semantic routing to direct queries to the right knowledge domain
- Hybrid search for better accuracy on technical terms
- Query decomposition for complex questions

## Local Mode of Qdrant

To use a local mode of Qdrant, you can specify the path to the database using the `--qdrant-local-path` argument:

```json
{
  "qdrant": {
    "command": "uvx",
    "args": [
      "mcp-server-qdrant",
      "--qdrant-local-path",
      "/path/to/qdrant/database",
      "--collection-name",
      "your_collection_name"
    ]
  }
}
```

It will run Qdrant local mode inside the same process as the MCP server. Although it is not recommended for production.

## Contributing

If you have suggestions for how mcp-server-qdrant could be improved, or want to report a bug, open an issue!
We'd love all and any contributions.

### Testing `mcp-server-qdrant` locally

The [MCP inspector](https://github.com/modelcontextprotocol/inspector) is a developer tool for testing and debugging MCP
servers. It runs both a client UI (default port 5173) and an MCP proxy server (default port 3000). Open the client UI in
your browser to use the inspector.

```shell
npx @modelcontextprotocol/inspector uv run mcp-server-qdrant \
  --collection-name test \
  --qdrant-local-path /tmp/qdrant-local-test
```

Once started, open your browser to http://localhost:5173 to access the inspector interface.

## License

This MCP server is licensed under the Apache License 2.0. This means you are free to use, modify, and distribute the
software, subject to the terms and conditions of the Apache License 2.0. For more details, please see the LICENSE file
in the project repository.
