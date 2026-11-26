import http.server
import json
import threading
import uuid
from unittest.mock import AsyncMock, patch

import pytest

from mcp_server_qdrant.qdrant import QdrantConnector, add_default_port


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        # A path prefix without a port gets the scheme's default port.
        ("https://host.com/qdrant", "https://host.com:443/qdrant"),
        ("http://host.com/qdrant", "http://host.com:80/qdrant"),
        ("https://host.com/a/b/", "https://host.com:443/a/b/"),
        ("http://[::1]/qdrant", "http://[::1]:80/qdrant"),
        # An explicit port is kept.
        ("https://host.com:8443/qdrant", "https://host.com:8443/qdrant"),
        # Without a path, the Qdrant client's 6333 default still applies.
        ("http://localhost:6333", "http://localhost:6333"),
        ("http://localhost", "http://localhost"),
        ("https://host.com/", "https://host.com/"),
        # Not a URL at all.
        (":memory:", ":memory:"),
    ],
)
def test_add_default_port(url, expected):
    assert add_default_port(url) == expected


class TestUrlParsing:
    """Test the location the connector passes to the Qdrant client."""

    @patch("mcp_server_qdrant.qdrant.AsyncQdrantClient")
    def test_url_with_path_prefix(self, mock_client_cls):
        """A URL with a path prefix gets the scheme's default port."""
        QdrantConnector(
            qdrant_url="https://host.com/qdrant",
            qdrant_api_key=None,
            collection_name="test",
            embedding_provider=AsyncMock(),
        )

        mock_client_cls.assert_called_once_with(
            location="https://host.com:443/qdrant", api_key=None, path=None
        )

    @patch("mcp_server_qdrant.qdrant.AsyncQdrantClient")
    def test_url_without_path(self, mock_client_cls):
        """A URL without a path is passed through unchanged."""
        QdrantConnector(
            qdrant_url="http://localhost:6333",
            qdrant_api_key=None,
            collection_name="test",
            embedding_provider=AsyncMock(),
        )

        mock_client_cls.assert_called_once_with(
            location="http://localhost:6333", api_key=None, path=None
        )

    @patch("mcp_server_qdrant.qdrant.AsyncQdrantClient")
    def test_url_with_custom_port_and_prefix(self, mock_client_cls):
        """A URL with an explicit port and a path prefix is passed through unchanged."""
        QdrantConnector(
            qdrant_url="https://host.com:8443/qdrant",
            qdrant_api_key=None,
            collection_name="test",
            embedding_provider=AsyncMock(),
        )

        mock_client_cls.assert_called_once_with(
            location="https://host.com:8443/qdrant", api_key=None, path=None
        )

    @patch("mcp_server_qdrant.qdrant.AsyncQdrantClient")
    def test_in_memory_location(self, mock_client_cls):
        QdrantConnector(
            qdrant_url=":memory:",
            qdrant_api_key=None,
            collection_name="test",
            embedding_provider=AsyncMock(),
        )

        mock_client_cls.assert_called_once_with(
            location=":memory:", api_key=None, path=None
        )

    @patch("mcp_server_qdrant.qdrant.AsyncQdrantClient")
    def test_local_path(self, mock_client_cls):
        QdrantConnector(
            qdrant_url=None,
            qdrant_api_key=None,
            collection_name="test",
            embedding_provider=AsyncMock(),
            qdrant_local_path="/tmp/qdrant",
        )

        mock_client_cls.assert_called_once_with(
            location=None, api_key=None, path="/tmp/qdrant"
        )


class _ProxyHandler(http.server.BaseHTTPRequestHandler):
    """Stands in for a reverse proxy: records each path and answers 404."""

    paths: list[str] = []

    def do_GET(self):
        self.paths.append(self.path)
        body = json.dumps({"status": {"error": "Not found"}, "time": 0}).encode()
        self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


# The client checks the server version in the background, and may only get to it
# after the fake proxy has shut down.
@pytest.mark.filterwarnings("ignore:Failed to obtain server version")
async def test_requests_go_to_the_path_prefix():
    """The prefix is kept on the requests that actually reach the proxy."""
    server = http.server.HTTPServer(("127.0.0.1", 0), _ProxyHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    _ProxyHandler.paths = []
    collection_name = f"prefix_{uuid.uuid4().hex}"
    try:
        port = server.server_address[1]
        connector = QdrantConnector(
            qdrant_url=f"http://127.0.0.1:{port}/qdrant",
            qdrant_api_key=None,
            collection_name=collection_name,
            embedding_provider=AsyncMock(),
        )
        results = await connector.search("hello")
    finally:
        server.shutdown()
        server.server_close()

    assert list(results) == []
    # The client's version check requests the prefix itself, "/qdrant", if it
    # arrives before the proxy shuts down.
    assert all(
        path == "/qdrant" or path.startswith("/qdrant/") for path in _ProxyHandler.paths
    )
    assert f"/qdrant/collections/{collection_name}" in _ProxyHandler.paths
