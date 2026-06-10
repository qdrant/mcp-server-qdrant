from mcp_server_qdrant.qdrant import (
    QdrantConnector,
    build_qdrant_client_url_options,
)


def test_qdrant_url_without_path_uses_location():
    assert build_qdrant_client_url_options("https://example.com:6333") == {
        "location": "https://example.com:6333"
    }


def test_qdrant_url_with_path_prefix_uses_url_and_prefix():
    assert build_qdrant_client_url_options("https://example.com/qdrant") == {
        "url": "https://example.com",
        "prefix": "/qdrant",
        "port": 443,
    }


def test_qdrant_http_url_with_path_prefix_uses_http_port():
    assert build_qdrant_client_url_options("http://example.com/qdrant") == {
        "url": "http://example.com",
        "prefix": "/qdrant",
        "port": 80,
    }


def test_qdrant_url_with_port_and_nested_path_prefix():
    assert build_qdrant_client_url_options(
        "http://example.com:8443/vector/qdrant/"
    ) == {
        "url": "http://example.com:8443",
        "prefix": "/vector/qdrant",
    }


def test_qdrant_non_http_location_is_preserved():
    assert build_qdrant_client_url_options(":memory:") == {"location": ":memory:"}


def test_connector_passes_reverse_proxy_prefix_to_client(monkeypatch):
    captured_options = {}

    class StubQdrantClient:
        def __init__(self, **kwargs):
            captured_options.update(kwargs)

    monkeypatch.setattr(
        "mcp_server_qdrant.qdrant.AsyncQdrantClient", StubQdrantClient
    )

    QdrantConnector(
        qdrant_url="https://example.com/qdrant",
        qdrant_api_key="secret",
        collection_name="memories",
        embedding_provider=object(),
    )

    assert captured_options == {
        "url": "https://example.com",
        "prefix": "/qdrant",
        "port": 443,
        "api_key": "secret",
        "path": None,
    }
