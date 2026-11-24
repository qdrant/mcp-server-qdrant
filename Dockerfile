FROM python:3.13-slim

# uv installs the exact dependency versions from uv.lock.
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /bin/uv

# Use the image's Python instead of letting uv download another one.
ENV UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Install the dependencies first, so code changes don't reinstall them.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project --no-cache

# Install the server itself from this repository's code.
COPY src ./src
RUN uv sync --locked --no-dev --no-editable --no-cache

ENV PATH="/app/.venv/bin:$PATH"

# Run as an unprivileged user. /data is where local mode can keep its storage
# (QDRANT_LOCAL_PATH=/data); a volume mounted there inherits the user's ownership.
RUN useradd --create-home --uid 1000 mcp \
    && mkdir /data \
    && chown mcp:mcp /data
USER mcp

# Listen on all interfaces so the server is reachable from outside the container.
ENV FASTMCP_HOST="0.0.0.0"

# Defaults that can be overridden at runtime. QDRANT_URL and QDRANT_API_KEY are
# left unset on purpose: an empty value counts as set and blocks QDRANT_LOCAL_PATH.
ENV COLLECTION_NAME="default-collection"
ENV EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"

# Default port for the SSE transport
EXPOSE 8000

# For stdio clients, override the command:
#   docker run -i --rm mcp-server-qdrant mcp-server-qdrant --transport stdio
CMD ["mcp-server-qdrant", "--transport", "sse"]
