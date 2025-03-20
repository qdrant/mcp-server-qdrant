#!/bin/bash
cd /Users/squishy64/mcp-server-qdrant
# Activate the virtual environment
source .venv/bin/activate
# Install the package in development mode
uv install -e .
