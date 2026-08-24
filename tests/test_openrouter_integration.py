import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_server_qdrant.embeddings.openrouter import OpenRouterEmbeddingProvider, OpenRouterEmbeddingSettings


@pytest.fixture
def mock_embedding_settings():
    """Create a mock embedding settings object."""
    return OpenRouterEmbeddingSettings(
        api_key="test_api_key",
        model_name="text-embedding-3-small"
    )


@pytest.fixture
def mock_embedding_provider(mock_embedding_settings):
    """Create a mock embedding provider."""
    return OpenRouterEmbeddingProvider(mock_embedding_settings)


@pytest.mark.asyncio
async def test_openrouter_provider_initialization():
    """Test that OpenRouter provider initializes correctly."""
    settings = OpenRouterEmbeddingSettings(
        api_key="test_api_key",
        model_name="text-embedding-3-small"
    )

    provider = OpenRouterEmbeddingProvider(settings)

    assert provider.settings.api_key == "test_api_key"
    assert provider.settings.model_name == "text-embedding-3-small"
    assert provider.model_name == "text-embedding-3-small"


@pytest.mark.asyncio
async def test_openrouter_provider_vector_name():
    """Test that OpenRouter provider generates correct vector names."""
    settings = OpenRouterEmbeddingSettings(
        api_key="test_api_key",
        model_name="text-embedding-3-small"
    )

    provider = OpenRouterEmbeddingProvider(settings)

    vector_name = provider.get_vector_name()
    assert vector_name == "openrouter-text-embedding-3-small"


@pytest.mark.asyncio
async def test_openrouter_provider_vector_size():
    """Test that OpenRouter provider returns correct vector sizes."""
    # Test small model
    settings_small = OpenRouterEmbeddingSettings(
        api_key="test_api_key",
        model_name="text-embedding-3-small"
    )

    provider_small = OpenRouterEmbeddingProvider(settings_small)
    assert provider_small.get_vector_size() == 1536

    # Test large model
    settings_large = OpenRouterEmbeddingSettings(
        api_key="test_api_key",
        model_name="text-embedding-3-large"
    )

    provider_large = OpenRouterEmbeddingProvider(settings_large)
    assert provider_large.get_vector_size() == 3072

    # Test default case (unknown model without "large" in name)
    settings_default = OpenRouterEmbeddingSettings(
        api_key="test_api_key",
        model_name="some-other-model"
    )

    provider_default = OpenRouterEmbeddingProvider(settings_default)
    assert provider_default.get_vector_size() == 1536


@pytest.mark.asyncio
async def test_openrouter_provider_vector_size_env_override():
    """Test that OPENROUTER_EMBEDDING_SIZE env var overrides the default size."""
    settings = OpenRouterEmbeddingSettings(
        api_key="test_api_key",
        model_name="some-custom-model"
    )
    provider = OpenRouterEmbeddingProvider(settings)

    with patch.dict(os.environ, {"OPENROUTER_EMBEDDING_SIZE": "2048"}):
        assert provider.get_vector_size() == 2048


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_openrouter_provider_embed_documents(mock_client_class):
    """Test that OpenRouter provider can embed documents."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "data": [
            {"embedding": [0.1, 0.2, 0.3]},
            {"embedding": [0.4, 0.5, 0.6]}
        ]
    }

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

    settings = OpenRouterEmbeddingSettings(
        api_key="test_api_key",
        model_name="text-embedding-3-small"
    )

    provider = OpenRouterEmbeddingProvider(settings)

    documents = ["Hello world", "Test document"]
    result = await provider.embed_documents(documents)

    assert len(result) == 2
    assert result[0] == [0.1, 0.2, 0.3]
    assert result[1] == [0.4, 0.5, 0.6]


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_openrouter_provider_embed_query(mock_client_class):
    """Test that OpenRouter provider can embed a query."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "data": [
            {"embedding": [0.1, 0.2, 0.3]}
        ]
    }

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

    settings = OpenRouterEmbeddingSettings(
        api_key="test_api_key",
        model_name="text-embedding-3-small"
    )

    provider = OpenRouterEmbeddingProvider(settings)

    query = "Test query"
    result = await provider.embed_query(query)

    assert len(result) == 3
    assert result == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_openrouter_provider_no_api_key():
    """Test that OpenRouter provider raises error when no API key is provided."""
    settings = OpenRouterEmbeddingSettings(
        api_key=None,
        model_name="text-embedding-3-small"
    )

    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("OPENROUTER_API_KEY", None)
        with pytest.raises(ValueError, match="OpenRouter API key is required"):
            OpenRouterEmbeddingProvider(settings)
