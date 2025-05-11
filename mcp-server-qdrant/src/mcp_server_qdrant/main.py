import argparse
import sys

print(">>> [TRACE] main.py: module loaded", file=sys.stderr, flush=True)

try:
    print(">>> [TRACE] main.py: imports complete", file=sys.stderr, flush=True)

    def main():
        print(">>> [TRACE] main.py: entered main()", file=sys.stderr, flush=True)
        # Parse the command-line arguments to determine the transport protocol.
        parser = argparse.ArgumentParser(description="mcp-server-qdrant")
        parser.add_argument(
            "--transport",
            choices=["stdio", "sse"],
            default="stdio",
        )
        args = parser.parse_args()

        print(">>> [TRACE] main.py: parsed args", file=sys.stderr, flush=True)

        # Import is done here to make sure environment variables are loaded
        # only after we make the changes.
        print(">>> [TRACE] main.py: about to import mcp_server_qdrant.server", file=sys.stderr, flush=True)
        from mcp_server_qdrant.server import mcp
        print(">>> [TRACE] main.py: imported mcp_server_qdrant.server", file=sys.stderr, flush=True)

        print(f">>> [TRACE] main.py: about to run mcp.run with transport={args.transport}", file=sys.stderr, flush=True)
        mcp.run(transport=args.transport)
        print(">>> [TRACE] main.py: mcp.run completed", file=sys.stderr, flush=True)

    if __name__ == "__main__":
        print(">>> [TRACE] main.py: __name__ == __main__", file=sys.stderr, flush=True)
        main()
        print(">>> [TRACE] main.py: main() completed", file=sys.stderr, flush=True)

except Exception as e:
    print(f">>> [TRACE] main.py: exception occurred: {e}", file=sys.stderr, flush=True)
    import traceback
    traceback.print_exc(file=sys.stderr)
    raise
