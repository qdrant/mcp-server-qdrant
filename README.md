# mcp-server-qdrant: A Qdrant MCP server


[![smithery badge](https://smithery.ai/badge/mcp-server-qdrant)](https://smithery.ai/protocol/mcp-server-qdrant)


> The [Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction) is an open protocol that enables
> seamless integration between LLM applications and external data sources and tools. Whether you're building an
> AI-powered IDE, enhancing a chat interface, or creating custom AI workflows, MCP provides a standardized way to
> connect LLMs with the context they need.


This repository is an example of how to create a MCP server for [Qdrant](https://qdrant.tech/), a vector search engine.


## Overview


An official Model Context Protocol server for keeping and retrieving memories in the Qdrant vector search engine.
It acts as a semantic memory layer on top of the Qdrant database.


## Components


### Tools


1. `qdrant-store`
   - Store some information in the Qdrant database
   - Input:
     - `information` (string): Information to store
     - `metadata` (JSON): Optional metadata to store
     - `collection_name` (string): Name of the collection to store the information in. This field is required if there are no default collection name.
                                   If there is a default collection name, this field is not enabled.
   - Returns: Confirmation message
2. `qdrant-find`
   - Retrieve relevant information from the Qdrant database
   - Input:
     - `query` (string): Query to use for searching
     - `collection_name` (string): Name of the collection to store the information in. This field is required if there are no default collection name.
                                   If there is a default collection name, this field is not enabled.
   - Returns: Information stored in the Qdrant database as separate messages


## Environment Variables


The configuration of the server is done using environment variables:


| Name                     | Description                                                         | Default Value                                                     |
|--------------------------|---------------------------------------------------------------------|-------------------------------------------------------------------|
| `QDRANT_URL`             | URL of the Qdrant server                                            | None                                                              |
| `QDRANT_API_KEY`         | API key for the Qdrant server                                       | None                                                              |
| `COLLECTION_NAME`        | Name of the default collection to use.                              | None                                                              |
| `QDRANT_LOCAL_PATH`      | Path to the local Qdrant database (alternative to `QDRANT_URL`)     | None                                                              |
| `EMBEDDING_PROVIDER`     | Embedding provider to use (`fastembed` or `openai`) | `fastembed`                                                       |
| `EMBEDDING_MODEL`        | Name of the embedding model to use                                  | `sentence-transformers/all-MiniLM-L6-v2`                          |
| `OPENAI_BASE_URL`       | Base URL for OpenAI-compatible API (required for `openai` provider) | None                                              |
| `OPENAI_API_KEY`        | API key for OpenAI-compatible API                                   | `not-needed`                                      |
| `TOOL_STORE_DESCRIPTION` | Custom description for the store tool                               | See default in [`settings.py`](src/mcp_server_qdrant/settings.py) |
| `TOOL_FIND_DESCRIPTION`  | Custom description for the find tool                                | See default in [`settings.py`](src/mcp_server_qdrant/settings.py) |


Note: You cannot provide both `QDRANT_URL` and `QDRANT_LOCAL_PATH` at the same time.


> [!IMPORTANT]
> Command-line arguments are not supported anymore! Please use environment variables for all configuration.


### FastMCP Environment Variables


Since `mcp-server-qdrant` is based on FastMCP, it also supports all the FastMCP environment variables. The most
important ones are listed below:


| Environment Variable                  | Description                                               | Default Value |
|---------------------------------------|-----------------------------------------------------------|---------------|
| `FASTMCP_DEBUG`                       | Enable debug mode                                         | `false`       |
| `FASTMCP_LOG_LEVEL`                   | Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) | `INFO`        |
| `FASTMCP_HOST`                        | Host address to bind the server to                        | `127.0.0.1`   |
| `FASTMCP_PORT`                        | Port to run the server on                                 | `8000`        |
| `FASTMCP_WARN_ON_DUPLICATE_RESOURCES` | Show warnings for duplicate resources                     | `true`        |
| `FASTMCP_WARN_ON_DUPLICATE_TOOLS`     | Show warnings for duplicate tools                         | `true`        |
