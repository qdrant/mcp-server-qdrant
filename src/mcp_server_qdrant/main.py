import argparse
import os


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

    # Import is done here to make sure environment variables are loaded
    # only after we make the changes.
    from mcp_server_qdrant.server import mcp

    run_kwargs = {}
    if args.transport != "stdio":
        # Bearer-token auth only makes sense for the HTTP-based transports.
        # If MCP_API_KEY isn't set, the server stays open (no auth check).
        api_key = os.environ.get("MCP_API_KEY")
        if api_key:
            from starlette.middleware import Middleware

            from mcp_server_qdrant.auth import BearerAuthMiddleware

            run_kwargs["middleware"] = [
                Middleware(BearerAuthMiddleware, api_key=api_key)
            ]

    mcp.run(transport=args.transport, **run_kwargs)
