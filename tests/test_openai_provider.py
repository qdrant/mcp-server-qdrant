import os
import pytest
from unittest.mock import patch, MagicMock

from mcp_server_qdrant.embeddings.openai import OpenAIEmbeddingProvider


class TestOpenAIProviderUnit:
    """Unit tests for OpenAIEmbeddingProvider."""

    def test_initialization(self):
        """Test that the provider can be initialized with a valid model."""
        provider = OpenAIEmbeddingProvider("text-embedding-3-small", "test-api-key")
        assert provider.model_name == "text-embedding-3-small"
        assert provider.api_key == "test-api-key"

    def test_invalid_model(self):
        """Test that initialization with an invalid model raises an error."""
        with pytest.raises(ValueError, match="Unsupported model"):
            OpenAIEmbeddingProvider("invalid-model", "test-api-key")

    def test_get_vector_name(self):
        """Test that the vector name is generated correctly."""
        provider = OpenAIEmbeddingProvider("text-embedding-3-small", "test-api-key")
        vector_name = provider.get_vector_name()
        assert vector_name == "openai-text-embedding-3-small"

    def test_get_vector_size(self):
        """Test that the vector size is correct for different models."""
        provider1 = OpenAIEmbeddingProvider("text-embedding-3-small", "test-api-key")
        assert provider1.get_vector_size() == 1536

        provider2 = OpenAIEmbeddingProvider("text-embedding-3-large", "test-api-key")
        assert provider2.get_vector_size() == 3072

        provider3 = OpenAIEmbeddingProvider("text-embedding-ada-002", "test-api-key")
        assert provider3.get_vector_size() == 1536

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient.post")
    async def test_embed_documents(self, mock_post):
        """Test that documents can be embedded."""
        # Mock response
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"index": 0, "embedding": [0.1, 0.2, 0.3]},
                {"index": 1, "embedding": [0.4, 0.5, 0.6]},
            ]
        }
        mock_post.return_value = mock_response

        provider = OpenAIEmbeddingProvider("text-embedding-3-small", "test-api-key")
        documents = ["Test document 1", "Test document 2"]
        embeddings = await provider.embed_documents(documents)

        # Check the API call
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer test-api-key"
        assert kwargs["json"]["model"] == "text-embedding-3-small"
        assert kwargs["json"]["input"] == documents

        # Check the result
        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient.post")
    async def test_embed_query(self, mock_post):
        """Test that a query can be embedded."""
        # Mock response
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"index": 0, "embedding": [0.1, 0.2, 0.3]},
            ]
        }
        mock_post.return_value = mock_response

        provider = OpenAIEmbeddingProvider("text-embedding-3-small", "test-api-key")
        query = "Test query"
        embedding = await provider.embed_query(query)

        # Check the API call
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert kwargs["json"]["input"] == [query]

        # Check the result
        assert embedding == [0.1, 0.2, 0.3]