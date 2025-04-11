# mcp-server-qdrant: A Qdrant MCP server

[![smithery badge](https://smithery.ai/badge/mcp-server-qdrant)](https://smithery.ai/protocol/mcp-server-qdrant)

> The [Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction) is an open protocol that enables
> seamless integration between LLM applications and external data sources and tools. Whether you're building an
> AI-powered IDE, enhancing a chat interface, or creating custom AI workflows, MCP provides a standardized way to
> connect LLMs with the context they need.

This repository is an example of how to create a MCP server for [Qdrant](https://qdrant.tech/), a vector search engine.

<a href="https://glama.ai/mcp/servers/9ejy5scw5i"><img width="380" height="200" src="https://glama.ai/mcp/servers/9ejy5scw5i/badge" alt="mcp-server-qdrant MCP server" /></a>

## Overview

An official Model Context Protocol server for keeping and retrieving memories in the Qdrant vector search engine. It
acts as a semantic memory layer on top of the Qdrant database.

The server can operate in two modes:

1. **Default Collection Mode** (`DefaultCollectionQdrantMCPServer`): When `COLLECTION_NAME` is specified, the server
   operates with a single, default collection. In this mode, the tools don't require collection names as parameters,
   making it simpler to use but limited to one collection.

1. **Multi-Collection Mode** (`MultiCollectionQdrantMCPServer`): When `COLLECTION_NAME` is not specified, the server
   allows working with multiple collections. In this mode:

   - Tools require `collection_name` as a parameter
   - Additional tools `qdrant-list-collections` and `qdrant-create-collection` are available
   - Useful for scenarios where different types of data need to be stored separately

## Components

### Tools

| Tool Name                  | Default Collection Mode | Multi-Collection Mode | Description                                     | Input Parameters                                                                                                                                                                  | Returns                                   |
| -------------------------- | :---------------------: | :-------------------: | ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| `qdrant-store`             |            ✓            |           ✓           | Store information in the Qdrant database        | - `information` (string): Information to store<br>- `metadata` (JSON): Optional metadata to store<br>- `collection_name` (string): Collection name *(Multi-Collection Mode only)* | Confirmation message                      |
| `qdrant-find`              |            ✓            |           ✓           | Retrieve relevant information from the database | - `query` (string): Search query<br>- `collection_name` (string): Collection name *(Multi-Collection Mode only)*                                                                  | Matching information as separate messages |
| `qdrant-list-collections`  |            ✗            |           ✓           | List all collections in Qdrant database         | None                                                                                                                                                                              | List of available collections             |
| `qdrant-create-collection` |            ✗            |           ✓           | Create a new collection in Qdrant               | - `collection_name` (string): Name of collection to create<br>- `description` (string): Purpose description                                                                       | Confirmation message                      |

> [!NOTE] In Default Collection Mode, `collection_name` parameters are not required as the server uses the collection
> specified in the `COLLECTION_NAME` environment variable.

## Environment Variables

The configuration of the server is done using environment variables:

| Name                                 | Description                                                         | Default Value                                                     |
| ------------------------------------ | ------------------------------------------------------------------- | ----------------------------------------------------------------- |
| `QDRANT_URL`                         | URL of the Qdrant server                                            | None                                                              |
| `QDRANT_API_KEY`                     | API key for the Qdrant server                                       | None                                                              |
| `COLLECTION_NAME`                    | Name of the default collection to use.                              | None                                                              |
| `QDRANT_LOCAL_PATH`                  | Path to the local Qdrant database (alternative to `QDRANT_URL`)     | None                                                              |
| `QDRANT_READ_ONLY`                   | Enable read-only mode (disables write operations)                   | `false`                                                           |
| `EMBEDDING_PROVIDER`                 | Embedding provider to use (currently only "fastembed" is supported) | `fastembed`                                                       |
| `EMBEDDING_MODEL`                    | Name of the embedding model to use                                  | `sentence-transformers/all-MiniLM-L6-v2`                          |
| `TOOL_STORE_DESCRIPTION`             | Custom description for the store tool                               | See default in [`settings.py`](src/mcp_server_qdrant/settings.py) |
| `TOOL_FIND_DESCRIPTION`              | Custom description for the find tool                                | See default in [`settings.py`](src/mcp_server_qdrant/settings.py) |
| `TOOL_LIST_COLLECTIONS_DESCRIPTION`  | Custom description for the list collections tool                    | See default in [`settings.py`](src/mcp_server_qdrant/settings.py) |
| `TOOL_CREATE_COLLECTION_DESCRIPTION` | Custom description for the create collection tool                   | See default in [`settings.py`](src/mcp_server_qdrant/settings.py) |

Note: You cannot provide both `QDRANT_URL` and `QDRANT_LOCAL_PATH` at the same time.

> [!IMPORTANT] Command-line arguments are not supported anymore! Please use environment variables for all configuration.

## Read-Only Mode

The server can be configured to operate in read-only mode by setting the `QDRANT_READ_ONLY` environment variable to
`true`. In this mode:

- The `qdrant-store` tool is disabled
- In Multi-Collection Mode, the `qdrant-create-collection` tool is also disabled
- All write operations are prevented
- Only search and retrieval operations are allowed

This mode is useful when you want to provide search capabilities over an existing dataset without allowing
modifications. Example configuration:

```bash
QDRANT_URL="http://localhost:6333" \
QDRANT_READ_ONLY="true" \
COLLECTION_NAME="my-collection" \
uvx mcp-server-qdrant
```

## Installation

### Using uvx

When using [`uvx`](https://docs.astral.sh/uv/guides/tools/#running-tools) no specific installation is needed to directly
run *mcp-server-qdrant*.

```shell
QDRANT_URL="http://localhost:6333" \
COLLECTION_NAME="my-collection" \
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2" \
uvx mcp-server-qdrant
```

#### Transport Protocols

The server supports different transport protocols that can be specified using the `--transport` flag:

```shell
QDRANT_URL="http://localhost:6333" \
COLLECTION_NAME="my-collection" \
uvx mcp-server-qdrant --transport sse
```

Supported transport protocols:

- `stdio` (default): Standard input/output transport, might only be used by local MCP clients
- `sse`: Server-Sent Events transport, perfect for remote clients

The default transport is `stdio` if not specified.

### Using Docker

A Dockerfile is available for building and running the MCP server:

```bash
# Build the container
docker build -t mcp-server-qdrant .

# Run the container
docker run -p 8000:8000 \
  -e QDRANT_URL="http://your-qdrant-server:6333" \
  -e QDRANT_API_KEY="your-api-key" \
  -e COLLECTION_NAME="your-collection" \
  mcp-server-qdrant
```

### Installing via Smithery

To install Qdrant MCP Server for Claude Desktop automatically via
[Smithery](https://smithery.ai/protocol/mcp-server-qdrant):

```bash
npx @smithery/cli install mcp-server-qdrant --client claude
```

### Manual configuration of Claude Desktop

To use this server with the Claude Desktop app, add the following configuration to the "mcpServers" section of your
`claude_desktop_config.json`:

```json
{
  "qdrant": {
    "command": "uvx",
    "args": ["mcp-server-qdrant"],
    "env": {
      "QDRANT_URL": "https://xyz-example.eu-central.aws.cloud.qdrant.io:6333",
      "QDRANT_API_KEY": "your_api_key",
      "COLLECTION_NAME": "your-collection-name",
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
      "COLLECTION_NAME": "your-collection-name",
      "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2"
    }
  }
}
```

This MCP server will automatically create a collection with the specified name if it doesn't exist.

By default, the server will use the `sentence-transformers/all-MiniLM-L6-v2` embedding model to encode memories. For the
time being, only [FastEmbed](https://qdrant.github.io/fastembed/) models are supported.

## Support for other tools

This MCP server can be used with any MCP-compatible client. For example, you can use it with
[Cursor](https://docs.cursor.com/context/model-context-protocol), which provides built-in support for the Model Context
Protocol.

### Using with Cursor/Windsurf

You can configure this MCP server to work as a code search tool for Cursor or Windsurf by customizing the tool
descriptions:

```bash
QDRANT_URL="http://localhost:6333" \
COLLECTION_NAME="code-snippets" \
TOOL_STORE_DESCRIPTION="Store reusable code snippets for later retrieval. \
The 'information' parameter should contain a natural language description of what the code does, \
while the actual code should be included in the 'metadata' parameter as a 'code' property. \
The value of 'metadata' is a Python dictionary with strings as keys. \
Use this whenever you generate some code snippet." \
TOOL_FIND_DESCRIPTION="Search for relevant code snippets based on natural language descriptions. \
The 'query' parameter should describe what you're looking for, \
and the tool will return the most relevant code snippets. \
Use this when you need to find existing code snippets for reuse or reference." \
uvx mcp-server-qdrant --transport sse # Enable SSE transport
```

In Cursor/Windsurf, you can then configure the MCP server in your settings by pointing to this running server using SSE
transport protocol. The description on how to add an MCP server to Cursor can be found in the
[Cursor documentation](https://docs.cursor.com/context/model-context-protocol#adding-an-mcp-server-to-cursor). If you
are running Cursor/Windsurf locally, you can use the following URL:

```
http://localhost:8000/sse
```

> [!TIP] We suggest SSE transport as a preferred way to connect Cursor/Windsurf to the MCP server, as it can support
> remote connections. That makes it easy to share the server with your team or use it in a cloud environment.

This configuration transforms the Qdrant MCP server into a specialized code search tool that can:

1. Store code snippets, documentation, and implementation details
1. Retrieve relevant code examples based on semantic search
1. Help developers find specific implementations or usage patterns

You can populate the database by storing natural language descriptions of code snippets (in the `information` parameter)
along with the actual code (in the `metadata.code` property), and then search for them using natural language queries
that describe what you're looking for.

> [!NOTE] The tool descriptions provided above are examples and may need to be customized for your specific use case.
> Consider adjusting the descriptions to better match your team's workflow and the specific types of code snippets you
> want to store and retrieve.

**If you have successfully installed the `mcp-server-qdrant`, but still can't get it to work with Cursor, please
consider creating the [Cursor rules](https://docs.cursor.com/context/rules-for-ai) so the MCP tools are always used when
the agent produces a new code snippet.** You can restrict the rules to only work for certain file types, to avoid using
the MCP server for the documentation or other types of content.

### Using with Claude Code

You can enhance Claude Code's capabilities by connecting it to this MCP server, enabling semantic search over your
existing codebase.

#### Setting up mcp-server-qdrant

1. Add the MCP server to Claude Code:

   ```shell
   # Add mcp-server-qdrant configured for code search
   claude mcp add code-search \
   -e QDRANT_URL="http://localhost:6333" \
   -e COLLECTION_NAME="code-repository" \
   -e EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2" \
   -e TOOL_STORE_DESCRIPTION="Store code snippets with descriptions. The 'information' parameter should contain a natural language description of what the code does, while the actual code should be included in the 'metadata' parameter as a 'code' property." \
   -e TOOL_FIND_DESCRIPTION="Search for relevant code snippets using natural language. The 'query' parameter should describe the functionality you're looking for." \
   -- uvx mcp-server-qdrant
   ```

1. Verify the server was added:

   ```shell
   claude mcp list
   ```

#### Using Semantic Code Search in Claude Code

Tool descriptions, specified in `TOOL_STORE_DESCRIPTION` and `TOOL_FIND_DESCRIPTION`, guide Claude Code on how to use
the MCP server. The ones provided above are examples and may need to be customized for your specific use case. However,
Claude Code should be already able to:

1. Use the `qdrant-store` tool to store code snippets with descriptions.
1. Use the `qdrant-find` tool to search for relevant code snippets using natural language.

### Run MCP server in Development Mode

The MCP server can be run in development mode using the `mcp dev` command. This will start the server and open the MCP
inspector in your browser.

```shell
COLLECTION_NAME=mcp-dev mcp dev src/mcp_server_qdrant/server.py
```

## Contributing

If you have suggestions for how mcp-server-qdrant could be improved, or want to report a bug, open an issue! We'd love
all and any contributions.

### Testing `mcp-server-qdrant` locally

The [MCP inspector](https://github.com/modelcontextprotocol/inspector) is a developer tool for testing and debugging MCP
servers. It runs both a client UI (default port 5173) and an MCP proxy server (default port 3000). Open the client UI in
your browser to use the inspector.

```shell
QDRANT_URL=":memory:" COLLECTION_NAME="test" \
mcp dev src/mcp_server_qdrant/server.py
```

Once started, open your browser to http://localhost:5173 to access the inspector interface.

> [!NOTE] The `QDRANT_URL=":memory:"` environment variable is used to run Qdrant in-memory mode. This is useful for
> testing purposes, but not recommended for production use.

## License

This MCP server is licensed under the Apache License 2.0. This means you are free to use, modify, and distribute the
software, subject to the terms and conditions of the Apache License 2.0. For more details, please see the LICENSE file
in the project repository.
