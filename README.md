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
     - Additional optional filter arguments, only present when filtering is configured. See [Filtering](#filtering).
   - Returns: Information stored in the Qdrant database as separate messages

## Filtering

`qdrant-find` can narrow down its results using the metadata stored next to each
entry. Filtering is **off by default**: unless one of the settings below is
configured, `qdrant-find` does not even have a filter argument.

### How filters address metadata

`qdrant-store` saves every entry as a point payload with two keys:

```json
{
  "document": "Use uv to manage dependencies",
  "metadata": { "language": "python", "repo": "mcp-server-qdrant" }
}
```

Filters therefore always target a field *inside* `metadata`, such as
`metadata.language`. The server adds that prefix for you in the declared filters
mode described below, but you have to write it yourself when passing a raw
filter.

### Declared filters (`FILTERABLE_FIELDS`)

The recommended way to enable filtering is to declare which metadata fields
should be filterable. `FILTERABLE_FIELDS` takes a JSON array of field
definitions:

```json
[
  {
    "name": "language",
    "description": "Programming language of the snippet",
    "field_type": "keyword",
    "condition": "=="
  },
  {
    "name": "repo",
    "description": "Repository the snippet came from",
    "field_type": "keyword",
    "condition": "!="
  },
  {
    "name": "archived",
    "description": "Whether the snippet is archived",
    "field_type": "boolean",
    "condition": "==",
    "required": true
  }
]
```

For every declared field, the server:

- creates a payload index, so filtering does not degrade into a full scan, and
- adds one typed and described argument to `qdrant-find`, named after the field.
  Arguments are optional by default and are skipped when not provided.

Fields marked `"required": true` become mandatory arguments of `qdrant-find`.

A field declared **without** a `condition` is indexed but exposes **no**
argument. This is useful for pre-indexing a field that is only ever queried
through a raw filter (see the next section).

Supported combinations of `field_type` and `condition`:

| `field_type` | Supported `condition` values         | Notes                              |
|--------------|--------------------------------------|------------------------------------|
| `keyword`    | `==`, `!=`, `any`, `except`          | `any` and `except` take a list     |
| `integer`    | `==`, `!=`, `>`, `>=`, `<`, `<=`, `any`, `except` | `any` and `except` take a list |
| `float`      | `>`, `>=`, `<`, `<=`                 | Range comparisons only             |
| `boolean`    | `==`, `!=`                           |                                    |

With the configuration above, `qdrant-find` accepts `language`, `repo` and
`archived` arguments, so a client can request Python snippets, exclude entries
coming from a given repository, and skip archived notes in a single call.
`archived` is mandatory in this example because it is marked as required.

### Raw filters (`QDRANT_ALLOW_ARBITRARY_FILTER`)

Setting `QDRANT_ALLOW_ARBITRARY_FILTER` to `true` keeps the raw `query_filter`
argument on `qdrant-find`, which accepts a
[Qdrant filter](https://qdrant.tech/documentation/concepts/filtering/) as JSON.
It is passed to Qdrant unchanged, so field keys have to be written in full:

```json
{
  "must": [
    { "key": "metadata.language", "match": { "value": "python" } }
  ]
}
```

No argument is generated for you in this mode, so the filter has to be produced
by the client. It is off by default.

Declaring a field **with** a `condition` takes precedence over this setting: the
raw `query_filter` argument is replaced by the generated arguments. To combine
both, declare the extra fields in `FILTERABLE_FIELDS` without a `condition`, and
filter on them through `query_filter`.

### Payload indexes are created with the collection

Payload indexes are created together with the collection. A collection that
already existed before `FILTERABLE_FIELDS` was configured keeps working, but
filtering on the newly declared fields is not accelerated by an index. Recreate
the collection, or create the index from the Qdrant side, if that matters for
your workload.

## Environment Variables

Configuration is done via environment variables. The only command-line argument is `--transport`, used to select the [transport protocol](#transport-protocols).

> [!NOTE]
> You cannot provide both `QDRANT_URL` and `QDRANT_LOCAL_PATH` at the same time.

| Name                     | Description                                                         | Default Value                                                     |
|--------------------------|---------------------------------------------------------------------|-------------------------------------------------------------------|
| `QDRANT_URL`             | URL of the Qdrant server                                            | None                                                              |
| `QDRANT_API_KEY`         | API key for the Qdrant server                                       | None                                                              |
| `COLLECTION_NAME`        | Name of the default collection to use.                              | None                                                              |
| `QDRANT_LOCAL_PATH`      | Path to the local Qdrant database (alternative to `QDRANT_URL`)     | None                                                              |
| `EMBEDDING_PROVIDER`     | Embedding provider to use (currently only "fastembed" is supported) | `fastembed`                                                       |
| `EMBEDDING_MODEL`        | Name of the embedding model to use                                  | `sentence-transformers/all-MiniLM-L6-v2`                          |
| `TOOL_STORE_DESCRIPTION` | Custom description for the store tool                               | See default in [`settings.py`](src/mcp_server_qdrant/settings.py) |
| `TOOL_FIND_DESCRIPTION`  | Custom description for the find tool                                | See default in [`settings.py`](src/mcp_server_qdrant/settings.py) |
| `QDRANT_SEARCH_LIMIT`    | Maximum number of results to return from search                     | `10`                                                              |
| `QDRANT_READ_ONLY`       | Enable read-only mode (disables `qdrant-store` tool)                | `false`                                                           |
| `FILTERABLE_FIELDS`      | JSON array of metadata fields exposed as `qdrant-find` filter arguments, see [Filtering](#filtering) | None                                     |
| `QDRANT_ALLOW_ARBITRARY_FILTER` | Expose the raw Qdrant `query_filter` argument on `qdrant-find`, see [Filtering](#filtering) | `false`                            |

### FastMCP Environment Variables

Since `mcp-server-qdrant` is based on FastMCP, it also supports all the FastMCP environment variables. The most
important ones are listed below:

| Environment Variable                       | Description                                                     | Default Value |
|--------------------------------------------|-----------------------------------------------------------------|---------------|
| `FASTMCP_LOG_LEVEL`                        | Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)       | `INFO`        |
| `FASTMCP_SERVER_DEBUG`                     | Enable debug mode                                               | `false`       |
| `FASTMCP_SERVER_HOST`                      | Host address to bind the server to                              | `127.0.0.1`   |
| `FASTMCP_SERVER_PORT`                      | Port to run the server on                                       | `8000`        |
| `FASTMCP_SERVER_ON_DUPLICATE_RESOURCES`    | Behavior for duplicate resources (warn, error, replace, ignore) | `warn`        |
| `FASTMCP_SERVER_ON_DUPLICATE_TOOLS`        | Behavior for duplicate tools (warn, error, replace, ignore)     | `warn`        |
| `FASTMCP_SERVER_ON_DUPLICATE_PROMPTS`      | Behavior for duplicate prompts (warn, error, replace, ignore)   | `warn`        |
| `FASTMCP_SERVER_DEPENDENCIES`              | List of dependencies to install in the server environment       | `[]`          |

> [!NOTE]
> Server-specific settings use the `FASTMCP_SERVER_` prefix. This may change in future versions.

## Installation

### Using uvx

When using [`uvx`](https://docs.astral.sh/uv/guides/tools/#running-tools) no specific installation is needed to directly run *mcp-server-qdrant*.

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
- `streamable-http`: Streamable HTTP transport, perfect for remote clients, more recent than SSE

The default transport is `stdio` if not specified.

When SSE transport is used, the server will listen on the specified port and wait for incoming connections. The default
port is 8000, however it can be changed using the `FASTMCP_SERVER_PORT` environment variable.

```shell
QDRANT_URL="http://localhost:6333" \
COLLECTION_NAME="my-collection" \
FASTMCP_SERVER_PORT=1234 \
uvx mcp-server-qdrant --transport sse
```

### Using Docker

A Dockerfile is available for building and running the MCP server:

```bash
# Build the container
docker build -t mcp-server-qdrant .

# Run the container
docker run -p 8000:8000 \
  -e FASTMCP_SERVER_HOST="0.0.0.0" \
  -e QDRANT_URL="http://your-qdrant-server:6333" \
  -e QDRANT_API_KEY="your-api-key" \
  -e COLLECTION_NAME="your-collection" \
  mcp-server-qdrant
```

> [!TIP]
> Please note that we set `FASTMCP_SERVER_HOST="0.0.0.0"` to make the server listen on all network interfaces. This is
> necessary when running the server in a Docker container.

### Installing via Smithery

To install Qdrant MCP Server for Claude Desktop automatically via [Smithery](https://smithery.ai/protocol/mcp-server-qdrant):

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

By default, the server will use the `sentence-transformers/all-MiniLM-L6-v2` embedding model to encode memories.
For the time being, only [FastEmbed](https://qdrant.github.io/fastembed/) models are supported.

## Support for other tools

This MCP server can be used with any MCP-compatible client. For example, you can use it with
[Cursor](https://docs.cursor.com/context/model-context-protocol) and [VS Code](https://code.visualstudio.com/docs), which provide built-in support for the Model Context
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

In Cursor/Windsurf, you can then configure the MCP server in your settings by pointing to this running server using
SSE transport protocol. The description on how to add an MCP server to Cursor can be found in the [Cursor
documentation](https://docs.cursor.com/context/model-context-protocol#adding-an-mcp-server-to-cursor). If you are
running Cursor/Windsurf locally, you can use the following URL:

```
http://localhost:8000/sse
```

> [!TIP]
> We suggest SSE transport as a preferred way to connect Cursor/Windsurf to the MCP server, as it can support remote
> connections. That makes it easy to share the server with your team or use it in a cloud environment.

This configuration transforms the Qdrant MCP server into a specialized code search tool that can:

1. Store code snippets, documentation, and implementation details
2. Retrieve relevant code examples based on semantic search
3. Help developers find specific implementations or usage patterns

You can populate the database by storing natural language descriptions of code snippets (in the `information` parameter)
along with the actual code (in the `metadata.code` property), and then search for them using natural language queries
that describe what you're looking for.

> [!NOTE]
> The tool descriptions provided above are examples and may need to be customized for your specific use case. Consider
> adjusting the descriptions to better match your team's workflow and the specific types of code snippets you want to
> store and retrieve.

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

2. Verify the server was added:

    ```shell
    claude mcp list
    ```

#### Using Semantic Code Search in Claude Code

Tool descriptions, specified in `TOOL_STORE_DESCRIPTION` and `TOOL_FIND_DESCRIPTION`, guide Claude Code on how to use
the MCP server. The ones provided above are examples and may need to be customized for your specific use case. However,
Claude Code should be already able to:

1. Use the `qdrant-store` tool to store code snippets with descriptions.
2. Use the `qdrant-find` tool to search for relevant code snippets using natural language.

### Run MCP server in Development Mode

The MCP server can be run in development mode using the `mcp dev` command. This will start the server and open the MCP
inspector in your browser.

```shell
COLLECTION_NAME=mcp-dev fastmcp dev src/mcp_server_qdrant/server.py
```

### Using with VS Code

For one-click installation, click one of the install buttons below:

[![Install with UVX in VS Code](https://img.shields.io/badge/VS_Code-UVX-0098FF?style=flat-square&logo=visualstudiocode&logoColor=white)](https://insiders.vscode.dev/redirect/mcp/install?name=qdrant&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22mcp-server-qdrant%22%5D%2C%22env%22%3A%7B%22QDRANT_URL%22%3A%22%24%7Binput%3AqdrantUrl%7D%22%2C%22QDRANT_API_KEY%22%3A%22%24%7Binput%3AqdrantApiKey%7D%22%2C%22COLLECTION_NAME%22%3A%22%24%7Binput%3AcollectionName%7D%22%7D%7D&inputs=%5B%7B%22type%22%3A%22promptString%22%2C%22id%22%3A%22qdrantUrl%22%2C%22description%22%3A%22Qdrant+URL%22%7D%2C%7B%22type%22%3A%22promptString%22%2C%22id%22%3A%22qdrantApiKey%22%2C%22description%22%3A%22Qdrant+API+Key%22%2C%22password%22%3Atrue%7D%2C%7B%22type%22%3A%22promptString%22%2C%22id%22%3A%22collectionName%22%2C%22description%22%3A%22Collection+Name%22%7D%5D) [![Install with UVX in VS Code Insiders](https://img.shields.io/badge/VS_Code_Insiders-UVX-24bfa5?style=flat-square&logo=visualstudiocode&logoColor=white)](https://insiders.vscode.dev/redirect/mcp/install?name=qdrant&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22mcp-server-qdrant%22%5D%2C%22env%22%3A%7B%22QDRANT_URL%22%3A%22%24%7Binput%3AqdrantUrl%7D%22%2C%22QDRANT_API_KEY%22%3A%22%24%7Binput%3AqdrantApiKey%7D%22%2C%22COLLECTION_NAME%22%3A%22%24%7Binput%3AcollectionName%7D%22%7D%7D&inputs=%5B%7B%22type%22%3A%22promptString%22%2C%22id%22%3A%22qdrantUrl%22%2C%22description%22%3A%22Qdrant+URL%22%7D%2C%7B%22type%22%3A%22promptString%22%2C%22id%22%3A%22qdrantApiKey%22%2C%22description%22%3A%22Qdrant+API+Key%22%2C%22password%22%3Atrue%7D%2C%7B%22type%22%3A%22promptString%22%2C%22id%22%3A%22collectionName%22%2C%22description%22%3A%22Collection+Name%22%7D%5D&quality=insiders)

[![Install with Docker in VS Code](https://img.shields.io/badge/VS_Code-Docker-0098FF?style=flat-square&logo=visualstudiocode&logoColor=white)](https://insiders.vscode.dev/redirect/mcp/install?name=qdrant&config=%7B%22command%22%3A%22docker%22%2C%22args%22%3A%5B%22run%22%2C%22-p%22%2C%228000%3A8000%22%2C%22-i%22%2C%22--rm%22%2C%22-e%22%2C%22QDRANT_URL%22%2C%22-e%22%2C%22QDRANT_API_KEY%22%2C%22-e%22%2C%22COLLECTION_NAME%22%2C%22mcp-server-qdrant%22%5D%2C%22env%22%3A%7B%22QDRANT_URL%22%3A%22%24%7Binput%3AqdrantUrl%7D%22%2C%22QDRANT_API_KEY%22%3A%22%24%7Binput%3AqdrantApiKey%7D%22%2C%22COLLECTION_NAME%22%3A%22%24%7Binput%3AcollectionName%7D%22%7D%7D&inputs=%5B%7B%22type%22%3A%22promptString%22%2C%22id%22%3A%22qdrantUrl%22%2C%22description%22%3A%22Qdrant+URL%22%7D%2C%7B%22type%22%3A%22promptString%22%2C%22id%22%3A%22qdrantApiKey%22%2C%22description%22%3A%22Qdrant+API+Key%22%2C%22password%22%3Atrue%7D%2C%7B%22type%22%3A%22promptString%22%2C%22id%22%3A%22collectionName%22%2C%22description%22%3A%22Collection+Name%22%7D%5D) [![Install with Docker in VS Code Insiders](https://img.shields.io/badge/VS_Code_Insiders-Docker-24bfa5?style=flat-square&logo=visualstudiocode&logoColor=white)](https://insiders.vscode.dev/redirect/mcp/install?name=qdrant&config=%7B%22command%22%3A%22docker%22%2C%22args%22%3A%5B%22run%22%2C%22-p%22%2C%228000%3A8000%22%2C%22-i%22%2C%22--rm%22%2C%22-e%22%2C%22QDRANT_URL%22%2C%22-e%22%2C%22QDRANT_API_KEY%22%2C%22-e%22%2C%22COLLECTION_NAME%22%2C%22mcp-server-qdrant%22%5D%2C%22env%22%3A%7B%22QDRANT_URL%22%3A%22%24%7Binput%3AqdrantUrl%7D%22%2C%22QDRANT_API_KEY%22%3A%22%24%7Binput%3AqdrantApiKey%7D%22%2C%22COLLECTION_NAME%22%3A%22%24%7Binput%3AcollectionName%7D%22%7D%7D&inputs=%5B%7B%22type%22%3A%22promptString%22%2C%22id%22%3A%22qdrantUrl%22%2C%22description%22%3A%22Qdrant+URL%22%7D%2C%7B%22type%22%3A%22promptString%22%2C%22id%22%3A%22qdrantApiKey%22%2C%22description%22%3A%22Qdrant+API+Key%22%2C%22password%22%3Atrue%7D%2C%7B%22type%22%3A%22promptString%22%2C%22id%22%3A%22collectionName%22%2C%22description%22%3A%22Collection+Name%22%7D%5D&quality=insiders)

#### Manual Installation

Add the following JSON block to your User Settings (JSON) file in VS Code. You can do this by pressing `Ctrl + Shift + P` and typing `Preferences: Open User Settings (JSON)`.

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
      }
    ],
    "servers": {
      "qdrant": {
        "command": "uvx",
        "args": ["mcp-server-qdrant"],
        "env": {
          "QDRANT_URL": "${input:qdrantUrl}",
          "QDRANT_API_KEY": "${input:qdrantApiKey}",
          "COLLECTION_NAME": "${input:collectionName}"
        }
      }
    }
  }
}
```

Or if you prefer using Docker, add this configuration instead:

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
      }
    ],
    "servers": {
      "qdrant": {
        "command": "docker",
        "args": [
          "run",
          "-p", "8000:8000",
          "-i",
          "--rm",
          "-e", "QDRANT_URL",
          "-e", "QDRANT_API_KEY",
          "-e", "COLLECTION_NAME",
          "mcp-server-qdrant"
        ],
        "env": {
          "QDRANT_URL": "${input:qdrantUrl}",
          "QDRANT_API_KEY": "${input:qdrantApiKey}",
          "COLLECTION_NAME": "${input:collectionName}"
        }
      }
    }
  }
}
```

Alternatively, you can create a `.vscode/mcp.json` file in your workspace with the following content:

```json
{
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
    }
  ],
  "servers": {
    "qdrant": {
      "command": "uvx",
      "args": ["mcp-server-qdrant"],
      "env": {
        "QDRANT_URL": "${input:qdrantUrl}",
        "QDRANT_API_KEY": "${input:qdrantApiKey}",
        "COLLECTION_NAME": "${input:collectionName}"
      }
    }
  }
}
```

For workspace configuration with Docker, use this in `.vscode/mcp.json`:

```json
{
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
    }
  ],
  "servers": {
    "qdrant": {
      "command": "docker",
      "args": [
        "run",
        "-p", "8000:8000",
        "-i",
        "--rm",
        "-e", "QDRANT_URL",
        "-e", "QDRANT_API_KEY",
        "-e", "COLLECTION_NAME",
        "mcp-server-qdrant"
      ],
      "env": {
        "QDRANT_URL": "${input:qdrantUrl}",
        "QDRANT_API_KEY": "${input:qdrantApiKey}",
        "COLLECTION_NAME": "${input:collectionName}"
      }
    }
  }
}
```

## Contributing

If you have suggestions for how mcp-server-qdrant could be improved, or want to report a bug, open an issue!
We'd love all and any contributions.

### Testing `mcp-server-qdrant` locally

The [MCP inspector](https://github.com/modelcontextprotocol/inspector) is a developer tool for testing and debugging MCP
servers. It runs both a client UI (default port 5173) and an MCP proxy server (default port 3000). Open the client UI in
your browser to use the inspector.

```shell
QDRANT_URL=":memory:" COLLECTION_NAME="test" \
fastmcp dev src/mcp_server_qdrant/server.py
```

Once started, open your browser to http://localhost:5173 to access the inspector interface.

## License

This MCP server is licensed under the Apache License 2.0. This means you are free to use, modify, and distribute the
software, subject to the terms and conditions of the Apache License 2.0. For more details, please see the LICENSE file
in the project repository.

## Privacy and support

- [Privacy Policy](https://qdrant.tech/legal/privacy-policy/)
- [Support](https://github.com/qdrant/mcp-server-qdrant/issues)
