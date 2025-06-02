import unittest
from unittest.mock import AsyncMock, Mock, patch
import pytest

from mcp_server_qdrant.embeddings.google_genai import GoogleGenAIProvider


class TestGoogleGenAIProviderIntegration:
    """Integration tests for GoogleGenAIProvider."""

    def test_initialization_with_api_key(self):
        """Test that the provider can be initialized with an API key."""
        provider = GoogleGenAIProvider(
            model_name="text-embedding-004",
            api_key="test-api-key"
        )
        assert provider.model_name == "text-embedding-004"
        assert provider.api_key == "test-api-key"
        assert provider.use_vertex_ai is False

    def test_initialization_with_vertex_ai(self):
        """Test that the provider can be initialized for Vertex AI."""
        provider = GoogleGenAIProvider(
            model_name="text-embedding-004",
            project="test-project",
            location="us-central1",
            use_vertex_ai=True
        )
        assert provider.model_name == "text-embedding-004"
        assert provider.project == "test-project"
        assert provider.location == "us-central1"
        assert provider.use_vertex_ai is True

    @patch('mcp_server_qdrant.embeddings.google_genai.genai')
    async def test_embed_documents(self, mock_genai):
        """Test that documents can be embedded."""
        # Mock the client and response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.embeddings = [Mock(values=[0.1, 0.2, 0.3])]
        mock_client.models.embed_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client

        provider = GoogleGenAIProvider(api_key="test-api-key")
        documents = ["This is a test document."]

        embeddings = await provider.embed_documents(documents)

        # Check that we got the right embeddings
        assert len(embeddings) == 1
        assert embeddings[0] == [0.1, 0.2, 0.3]

        # Verify the client was called correctly
        mock_client.models.embed_content.assert_called_once()
        call_args = mock_client.models.embed_content.call_args
        assert call_args[1]['model'] == "text-embedding-004"
        assert call_args[1]['contents'] == "This is a test document."

    @patch('mcp_server_qdrant.embeddings.google_genai.genai')
    async def test_embed_query(self, mock_genai):
        """Test that queries can be embedded."""
        # Mock the client and response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.embeddings = [Mock(values=[0.1, 0.2, 0.3])]
        mock_client.models.embed_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client

        provider = GoogleGenAIProvider(api_key="test-api-key")
        query = "This is a test query."

        embedding = await provider.embed_query(query)

        # Check that we got the right embedding
        assert embedding == [0.1, 0.2, 0.3]

        # Verify the client was called correctly
        mock_client.models.embed_content.assert_called_once()
        call_args = mock_client.models.embed_content.call_args
        assert call_args[1]['model'] == "text-embedding-004"
        assert call_args[1]['contents'] == "This is a test query."

    def test_get_vector_name(self):
        """Test that the vector name is generated correctly."""
        provider = GoogleGenAIProvider(model_name="text-embedding-004")
        vector_name = provider.get_vector_name()

        # Check that the vector name follows the expected format
        assert vector_name == "google-genai-004"

    def test_get_vector_size_known_models(self):
        """Test vector sizes for known models."""
        test_cases = [
            ("text-embedding-004", 768),
            ("text-embedding-005", 768),
            ("text-multilingual-embedding-002", 768),
            ("gemini-embedding-001", 3072),
        ]

        for model_name, expected_size in test_cases:
            provider = GoogleGenAIProvider(model_name=model_name)
            assert provider.get_vector_size() == expected_size

    def test_get_vector_size_unknown_model(self):
        """Test vector size for unknown model defaults to 768."""
        provider = GoogleGenAIProvider(model_name="unknown-model")
        assert provider.get_vector_size() == 768

    def test_import_error_handling(self):
        """Test that ImportError is raised when google-genai is not available."""
        with patch('mcp_server_qdrant.embeddings.google_genai.genai', side_effect=ImportError()):
            provider = GoogleGenAIProvider(api_key="test-api-key")
            
            with pytest.raises(ImportError, match="google-genai package is required"):
                _ = provider.client 