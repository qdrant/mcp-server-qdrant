---
name: setup
description: Configure the Qdrant Power after installation. Use this skill for missing uvx, missing environment variables, unapproved environment variables, unavailable Qdrant tools, "Failed to connect" errors, and setup requests.
---

# Configure the Qdrant Power

After the Power is installed, guide the user through these steps.

Kiro IDE needs three separate things to be true before the server starts:

1. `uvx` is reachable from the process that launched Kiro.
2. The four variables exist in that process environment.
3. The four variable names appear in Kiro's approved list.

A failure in any one of them shows up as the same `Failed to connect` log line,
so work through the steps in order instead of guessing.

## 1. Make sure that uvx is available

Run this command:

```shell
uvx --version
```

If the command succeeds, note the full path:

```shell
command -v uvx
```

If the command fails, tell the user that the Power requires uv.

Before you run the installation command, ask the user for approval.

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After the installation, run `uvx --version` again.

If `uvx` is still unavailable, tell the user to restart the terminal and Kiro.

CAUTION: `uv` installs `uvx` into `~/.local/bin`. A Kiro IDE started from the
macOS Dock or Finder receives a minimal `PATH` of `/usr/bin:/bin:/usr/sbin:/sbin`
and cannot resolve the bare `uvx` command. Step 5 covers this.

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

## 3. Set the environment variables

Give the user this template with the selected values:

```shell
export QDRANT_URL="<qdrant-url>"
export QDRANT_API_KEY="<qdrant-api-key>"
export COLLECTION_NAME="<collection-name>"
export EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
```

Tell the user to put these lines in a shell startup file such as `~/.zshrc`, or
in a file that a startup file sources. Variables typed at a prompt live only in
that one terminal session.

Tell the user to keep the API key out of shell history by editing the file
directly instead of typing an `export` command at the prompt.

Tell Kiro CLI users to export these variables before they start Kiro.

Tell Kiro IDE users to set these variables in the environment that starts Kiro.

## 4. Approve the variables in Kiro IDE

Kiro IDE expands only approved environment variables in MCP config files. Until
the four names are approved, `${QDRANT_URL}` and the others stay unexpanded and
the server fails to start.

When Kiro detects unapproved variables in an MCP server configuration, it shows
a security warning that lists them. The user can approve them from that popup.

To manage the list manually, tell the user to:

1. Open Kiro settings.
2. Search for `Mcp Approved Env Vars`.
3. Add `QDRANT_URL`, `QDRANT_API_KEY`, `COLLECTION_NAME`, and `EMBEDDING_MODEL`.

This setting keeps MCP servers from reading arbitrary environment variables.

## 5. Restart Kiro

Tell the user to quit Kiro completely and start it again.

Reconnecting the MCP server is not enough. The parent Kiro process keeps the
environment it was launched with, so a server restart reuses the same empty
variables.

On macOS, tell the user to launch Kiro from a terminal that has the variables:

```shell
source ~/.zshrc
kiro
```

This also gives Kiro a full `PATH`, which lets it resolve `uvx`.

If the user prefers to launch Kiro from the Dock or Finder, the bare `uvx`
command will not resolve. Tell the user to set an absolute command path in the
Power's MCP server entry instead, using the path from step 1:

```json
"command": "/Users/<user>/.local/bin/uvx"
```

## 6. Make sure that the connection operates

After Qdrant reconnects, use `qdrant-find` with a harmless query.

If the tool returns no memories, report that the connection operates and the collection is empty.

If the tool returns an error, explain the error and repeat the applicable configuration step.

The first start downloads the embedding model, which can take several seconds
and may time out once. Tell the user to reconnect if that happens. Later starts
use the cached model.

## Troubleshooting `Failed to connect`

Kiro logs this line for every startup failure. Use these checks to tell the
causes apart.

Confirm the variables reached the Kiro process. Replace `<pid>` with the Kiro
process id from `pgrep -f "Kiro.app/Contents/MacOS"`:

```shell
ps eww -p <pid> | tr ' ' '\n' | grep -c '^QDRANT_'
```

A count of `0` means step 3, step 4, or step 5 is incomplete.

Confirm Kiro can resolve the command:

```shell
ps eww -p <pid> | tr ' ' '\n' | grep '^PATH='
```

A `PATH` of `/usr/bin:/bin:/usr/sbin:/sbin` means Kiro was launched from the
Dock or Finder and cannot find `uvx`. Apply step 5.

Confirm the credentials and URL independently of Kiro:

```shell
curl -s -o /dev/null -w '%{http_code}\n' -H "api-key: $QDRANT_API_KEY" "$QDRANT_URL/collections"
```

`200` means the URL and key are good and the fault is in the Kiro
configuration. Qdrant Cloud accepts the base URL with or without an explicit
`:6333` port.
