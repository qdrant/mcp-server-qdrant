import argparse
import os
import warnings

# FastMCP 2.x read these names. FastMCP 3 and later only read the new ones,
# so map the old names to the new ones to keep existing setups working.
DEPRECATED_ENV_VARS = {
    "FASTMCP_SERVER_HOST": "FASTMCP_HOST",
    "FASTMCP_SERVER_PORT": "FASTMCP_PORT",
    "FASTMCP_SERVER_DEBUG": "FASTMCP_DEBUG",
}


def apply_deprecated_env_vars() -> None:
    """
    Copy each deprecated FASTMCP_SERVER_* variable to its new name, unless the
    new name is already set. Must run before FastMCP is imported, because
    FastMCP reads its settings from the environment at import time.
    """
    for old_name, new_name in DEPRECATED_ENV_VARS.items():
        if old_name not in os.environ:
            continue
        if new_name not in os.environ:
            os.environ[new_name] = os.environ[old_name]
        # FutureWarning, unlike DeprecationWarning, is shown to end users by default.
        warnings.warn(
            f"{old_name} is deprecated and will be removed in a future release. "
            f"Use {new_name} instead.",
            FutureWarning,
            stacklevel=2,
        )


def main():
    """
    Main entry point for the mcp-server-qdrant script defined
    in pyproject.toml. It runs the MCP server with a specific transport
    protocol.
    """

    # Parse the command-line arguments to determine the transport protocol.
    parser = argparse.ArgumentParser(description="mcp-server-qdrant")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
    )
    args = parser.parse_args()

    apply_deprecated_env_vars()

    # Import is done here to make sure environment variables are loaded
    # only after we make the changes.
    from mcp_server_qdrant.server import mcp

    mcp.run(transport=args.transport)
