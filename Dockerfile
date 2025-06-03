# Multi-stage build for optimized image
FROM python:3.11-slim as builder

# Build arguments for version control
ARG PACKAGE_REF=main
ARG PACKAGE_REPO=https://github.com/AndrzejOlender/mcp-server-qdrant.git

WORKDIR /app

# Install git and uv for package management
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv

# Create a virtual environment
RUN uv venv /opt/venv

# Make sure we use the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Install the mcp-server-qdrant package from GitHub
RUN echo "Installing from: ${PACKAGE_REPO}@${PACKAGE_REF}" && \
    uv pip install --no-cache-dir "git+${PACKAGE_REPO}@${PACKAGE_REF}"

# Production stage
FROM python:3.11-slim as production

# Install git and curl (needed for potential reinstalls and health checks)
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Make sure we use the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Set up working directory
WORKDIR /app
RUN chown appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose the default port for SSE transport
EXPOSE 8000

# Set environment variables with defaults that can be overridden at runtime
ENV QDRANT_URL=""
ENV QDRANT_API_KEY=""
ENV COLLECTION_NAME="default-collection"
ENV EMBEDDING_PROVIDER="fastembed"
ENV EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
ENV GOOGLE_API_KEY=""
ENV GOOGLE_CLOUD_PROJECT=""
ENV GOOGLE_CLOUD_LOCATION="us-central1"
ENV GOOGLE_GENAI_USE_VERTEXAI="false"

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run the server with SSE transport
CMD ["mcp-server-qdrant", "--transport", "sse"]
