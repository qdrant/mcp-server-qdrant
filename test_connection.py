#!/usr/bin/env python
"""
A simple script to test if the MCP server is accepting connections.
"""
import subprocess
import time
import sys
import json

def main():
    # Start the MCP server in a subprocess
    print("Starting MCP server...")
    process = subprocess.Popen(
        ["mcp-server-qdrant"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    # Give it a moment to start up
    time.sleep(2)
    
    # Send a basic initialization message
    init_message = {
        "jsonrpc": "2.0",
        "method": "initialize",
        "params": {
            "clientName": "test-client",
            "clientVersion": "1.0.0",
        },
        "id": "1"
    }
    
    print("Sending initialization message...")
    process.stdin.write(json.dumps(init_message) + "\n")
    process.stdin.flush()
    
    # Wait for a response
    print("Waiting for response...")
    response = process.stdout.readline()
    
    if response:
        print(f"Received response: {response}")
        print("Server is responding to MCP requests!")
    else:
        print("No response received from server")
        # Check if there's any error output
        errors = process.stderr.read()
        if errors:
            print(f"Errors: {errors}")
    
    # Terminate the server
    print("Terminating server...")
    process.terminate()
    process.wait()
    
    print("Done")

if __name__ == "__main__":
    main()
