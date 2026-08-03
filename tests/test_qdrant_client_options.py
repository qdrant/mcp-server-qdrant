import pytest

from mcp_server_qdrant.qdrant import qdrant_client_options


def test_qdrant_url_without_prefix_uses_location():
    assert qdrant_client_options("https://qdrant.example.com:6333") == {
        "location": "https://qdrant.example.com:6333"
    }


def test_in_memory_qdrant_uses_location():
    assert qdrant_client_options(":memory:") == {"location": ":memory:"}


def test_qdrant_url_with_prefix_uses_base_url_and_prefix():
    assert qdrant_client_options("https://qdrant.example.com/qdrant/") == {
        "url": "https://qdrant.example.com",
        "prefix": "/qdrant",
    }


def test_qdrant_url_prefix_requires_absolute_url():
    with pytest.raises(ValueError, match="scheme and host"):
        qdrant_client_options("qdrant.example.com/qdrant")
