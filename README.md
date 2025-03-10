# mcp-server-qdrant: A Qdrant MCP server
[![smithery badge](https://smithery.ai/badge/mcp-server-qdrant)](https://smithery.ai/protocol/mcp-server-qdrant)

> The [Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction) is an open protocol that enables
> seamless integration between LLM applications and external data sources and tools. Whether you’re building an
> AI-powered IDE, enhancing a chat interface, or creating custom AI workflows, MCP provides a standardized way to
> connect LLMs with the context they need.

This repository is an example of how to create a MCP server for [Qdrant](https://qdrant.tech/), a vector search engine.

<a href="https://glama.ai/mcp/servers/9ejy5scw5i"><img width="380" height="200" src="https://glama.ai/mcp/servers/9ejy5scw5i/badge" alt="mcp-server-qdrant MCP server" /></a>

## Overview

A basic Model Context Protocol server for keeping and retrieving memories in the Qdrant vector search engine.
It acts as a semantic memory layer on top of the Qdrant database.

## Components

### Tools

1. `qdrant-store`
   - Store some information in the Qdrant database
   - Input:
     - `information` (string): Information to store
     - `metadata` (JSON): Optional metadata to store
   - Returns: Confirmation message
2. `qdrant-find`
   - Retrieve relevant information from the Qdrant database
   - Input:
     - `query` (string): Query to use for searching
   - Returns: Information stored in the Qdrant database as separate messages

## Installation in Claude Desktop

### Using mcp (recommended)

When using [`mcp`](https://github.com/modelcontextprotocol/python-sdk) no specific installation is needed to directly run *mcp-server-qdrant*.

```shell
mcp install src/mcp_server_qdrant/server.py \
  -v QDRANT_URL="http://localhost:6333" \
  -v QDRANT_API_KEY="your_api_key" \
  -v COLLECTION_NAME="my_collection" \
  -v EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
```

### Installing via Smithery

To install Qdrant MCP Server for Claude Desktop automatically via [Smithery](https://smithery.ai/protocol/mcp-server-qdrant):

```bash
npx @smithery/cli install mcp-server-qdrant --client claude
```

### Manual configuration

To use this server with the Claude Desktop app, add the following configuration to the "mcpServers" section of your
`claude_desktop_config.json`:

```json
{
  "qdrant": {
    "command": "uvx",
    "args": ["mcp-server-qdrant"],
    "env": {
      "QDRANT_URL": "http://localhost:6333",
      "QDRANT_API_KEY": "your_api_key",
      "COLLECTION_NAME": "your_collection_name",
      "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2"
    }
  }
}
```

For local Qdrant mode:

```json
{
  "qdrant": {
    "command": "uvx",
    "args": ["mcp-server-qdrant"],
    "env": {
      "QDRANT_LOCAL_PATH": "/path/to/qdrant/database",
      "COLLECTION_NAME": "your_collection_name",
      "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2"
    }
  }
}
```

This MCP server will automatically create a collection with the specified name if it doesn't exist.

By default, the server will use the `sentence-transformers/all-MiniLM-L6-v2` embedding model to encode memories.
For the time being, only [FastEmbed](https://qdrant.github.io/fastembed/) models are supported.

### Support for other tools

This MCP server can be used with any MCP-compatible client. For example, you can use it with
[Cursor](https://docs.cursor.com/context/model-context-protocol), which provides built-in support for the Model Context
Protocol.

## Environment Variables

The configuration of the server is done using environment variables:

- `QDRANT_URL`: URL of the Qdrant server, e.g. `http://localhost:6333`
- `QDRANT_API_KEY`: API key for the Qdrant server (optional, depends on Qdrant server configuration)
- `COLLECTION_NAME`: Name of the collection to use (required)
- `EMBEDDING_MODEL`: Name of the embedding model to use (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `EMBEDDING_PROVIDER`: Embedding provider to use (currently only "fastembed" is supported)
- `QDRANT_LOCAL_PATH`: Path to the local Qdrant database (alternative to `QDRANT_URL`)

Note: You cannot provide both `QDRANT_URL` and `QDRANT_LOCAL_PATH` at the same time.

> [!IMPORTANT]
> Command-line arguments are not supported anymore! Please use environment variables for all configuration.

## Contributing

If you have suggestions for how mcp-server-qdrant could be improved, or want to report a bug, open an issue!
We'd love all and any contributions.

### Testing `mcp-server-qdrant` locally

The [MCP inspector](https://github.com/modelcontextprotocol/inspector) is a developer tool for testing and debugging MCP
servers. It runs both a client UI (default port 5173) and an MCP proxy server (default port 3000). Open the client UI in
your browser to use the inspector.

```shell
QDRANT_URL=":memory:" COLLECTION_NAME="test" \
mcp dev src/mcp_server_qdrant/server.py
```

Once started, open your browser to http://localhost:5173 to access the inspector interface.

## License

This MCP server is licensed under the Apache License 2.0. This means you are free to use, modify, and distribute the
software, subject to the terms and conditions of the Apache License 2.0. For more details, please see the LICENSE file
in the project repository.
