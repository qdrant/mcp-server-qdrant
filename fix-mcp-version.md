# Fixing MCP Version Incompatibility

## Problem

There's a version mismatch between the specified MCP dependency and the installed version:

- **Specified in pyproject.toml**: `mcp==0.9.1`
- **Actually installed**: `mcp==1.4.1`

This has led to an import error because the code is looking for `mcp.server.context` which exists in version 0.9.1 but was moved to `mcp.server.fastmcp` in newer versions.

## Solution

1. We've updated the `pyproject.toml` file to specify `mcp>=1.4.1` instead of `mcp==0.9.1`
2. The next step is to reinstall the package with the updated dependencies

## How to Fix

Execute the following commands:

```bash
cd /Users/squishy64/mcp-server-qdrant
# Activate the virtual environment 
source .venv/bin/activate
# Reinstall with updated dependencies
pip install -e .
# Or using uv
uv install -e .
```

## Verification

After reinstalling, you should be able to run:

```bash
mcp-server-qdrant --collection-name test --qdrant-local-path /tmp/qdrant-test
```

Without seeing the "ModuleNotFoundError: No module named 'mcp.server.context'" error.
