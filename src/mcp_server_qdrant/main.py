from mcp_server_qdrant.server import mcp


def main():
    """
    Main entry point for the mcp-server-qdrant script defined
    in pyproject.toml. It runs the MCP server.
    """
    mcp.run()
