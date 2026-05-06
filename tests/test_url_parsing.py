from unittest.mock import AsyncMock, patch

import pytest

from mcp_server_qdrant.qdrant import QdrantConnector


class TestUrlParsing:
    """Test that Qdrant URLs are correctly parsed into location, port, and prefix."""

    @patch("mcp_server_qdrant.qdrant.AsyncQdrantClient")
    def test_url_with_path_prefix(self, mock_client_cls):
        """Test that a URL with a path prefix is parsed into location, port, and prefix."""
        mock_client_cls.return_value = AsyncMock()
        embedding_provider = AsyncMock()

        QdrantConnector(
            qdrant_url="https://host.com/qdrant",
            qdrant_api_key=None,
            collection_name="test",
            embedding_provider=embedding_provider,
        )

        mock_client_cls.assert_called_once_with(
            api_key=None,
            path=None,
            location="https://host.com",
            port=443,
            prefix="qdrant",
        )

    @patch("mcp_server_qdrant.qdrant.AsyncQdrantClient")
    def test_url_without_path(self, mock_client_cls):
        """Test that a URL without a path is passed directly as location."""
        mock_client_cls.return_value = AsyncMock()
        embedding_provider = AsyncMock()

        QdrantConnector(
            qdrant_url="http://localhost:6333",
            qdrant_api_key=None,
            collection_name="test",
            embedding_provider=embedding_provider,
        )

        mock_client_cls.assert_called_once_with(
            api_key=None,
            path=None,
            location="http://localhost:6333",
        )

    @patch("mcp_server_qdrant.qdrant.AsyncQdrantClient")
    def test_url_with_custom_port_and_prefix(self, mock_client_cls):
        """Test that a URL with an explicit port and path prefix extracts the port correctly."""
        mock_client_cls.return_value = AsyncMock()
        embedding_provider = AsyncMock()

        QdrantConnector(
            qdrant_url="https://host.com:8443/qdrant",
            qdrant_api_key=None,
            collection_name="test",
            embedding_provider=embedding_provider,
        )

        mock_client_cls.assert_called_once_with(
            api_key=None,
            path=None,
            location="https://host.com",
            port=8443,
            prefix="qdrant",
        )
