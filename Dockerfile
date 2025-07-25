FROM python:3.13-slim

WORKDIR /app

# Install uv for package management
COPY --from=ghcr.io/astral-sh/uv:0.8.3 /uv /uvx /bin/

# Install the mcp-server-qdrant package
RUN uv tool install --no-cache mcp-server-qdrant

# Expose the default port for SSE transport
EXPOSE 8000

# Set environment variables with defaults that can be overridden at runtime
ENV QDRANT_URL=""
ENV QDRANT_API_KEY=""
ENV COLLECTION_NAME="default-collection"
ENV EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"

# Make the server listen on all network interfaces
ENV FASTMCP_HOST="0.0.0.0"

# Run the server with SSE transport
CMD ["uvx", "mcp-server-qdrant", "--transport", "sse"]
