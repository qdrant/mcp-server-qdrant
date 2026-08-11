---
name: setup
description: Configure the Qdrant Power after installation. Use this skill for missing uvx, missing environment variables, unavailable Qdrant tools, and setup requests.
---

# Configure the Qdrant Power

After the Power is installed, guide the user through these steps.

## 1. Make sure that uvx is available

Run this command:

```shell
uvx --version
```

If the command succeeds, continue to the configuration.

If the command fails, tell the user that the Power requires uv.

Before you run the installation command, ask the user for approval.

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After the installation, run `uvx --version` again.

If `uvx` is still unavailable, tell the user to restart the terminal and Kiro.

## 2. Collect the configuration

Ask the user for these values:

- `QDRANT_URL`: The URL of the Qdrant server.
- `COLLECTION_NAME`: The collection for semantic memories.
- `EMBEDDING_MODEL`: The FastEmbed model name.

Use `http://localhost:6333` as the usual local URL.

Use the Qdrant Cloud URL for a cloud connection.

If the user does not select an embedding model, use `sentence-transformers/all-MiniLM-L6-v2`.

If the user uses Qdrant Cloud, tell the user to set `QDRANT_API_KEY` locally.

If the user uses local Qdrant without authentication, use an empty `QDRANT_API_KEY` value.

Do not ask the user to send an API key in the conversation.

Tell the user that the server creates the collection automatically.

CAUTION: Do not change the embedding model for an existing collection. A different vector size can cause store or search errors.

## 3. Configure Kiro

Give the user this template with the selected values:

```shell
export QDRANT_URL="<qdrant-url>"
export QDRANT_API_KEY="<qdrant-api-key>"
export COLLECTION_NAME="<collection-name>"
export EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
```

Tell Kiro CLI users to export these variables before they start Kiro.

Tell Kiro IDE users to set these variables in the environment that starts Kiro.

If Kiro requests access, tell Kiro IDE users to approve the variables.

After the user configures the variables, tell the user to restart Kiro or reconnect the Qdrant MCP server.

## 4. Make sure that the connection operates

After Qdrant reconnects, use `qdrant-find` with a harmless query.

If the tool returns no memories, report that the connection operates and the collection is empty.

If the tool returns an error, explain the error and repeat the applicable configuration step.
