FROM python:3.13-slim

WORKDIR /app

# Install uv for package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install the mcp-server-qdrant package
RUN uv tool install --no-cache mcp-server-qdrant

# Expose the default port for SSE transport
EXPOSE 8000

# Make the server listen on all network interfaces
ENV FASTMCP_HOST="0.0.0.0"

# Run the server with SSE transport
CMD ["uvx", "mcp-server-qdrant", "--transport", "sse"]
