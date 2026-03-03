import os
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from mcp_server_qdrant.embeddings.gemini import GeminiEmbeddingProvider

@pytest.mark.asyncio
class TestGeminiEmbeddingProviderUnit:
    """Unit tests for GeminiEmbeddingProvider with mocked API."""

    @pytest.fixture
    def mock_genai_client(self):
        """Mock genai.Client for testing without API calls."""
        with patch('mcp_server_qdrant.embeddings.gemini.genai.Client') as mock_client_class:
            mock_instance = MagicMock()
            mock_client_class.return_value = mock_instance
            
            # Mock async embed_content
            mock_embedding1 = MagicMock()
            mock_embedding1.values = [0.1] * 768
            mock_embedding2 = MagicMock()
            mock_embedding2.values = [0.2] * 768
            
            mock_response = MagicMock()
            mock_response.embeddings = [mock_embedding1, mock_embedding2]
            mock_instance.aio.models.embed_content = AsyncMock(return_value=mock_response)
            
            # Mock sync embed_content
            mock_instance.models.embed_content = MagicMock(return_value=mock_response)
            
            yield mock_instance

    @pytest.fixture
    def gemini_provider(self, mock_genai_client):
        """Create provider instance with mocked client."""
        return GeminiEmbeddingProvider(model_name="text-embedding-004", api_key="fake-key")

    async def test_initialization(self, gemini_provider):
        """Test provider setup, model name normalization."""
        assert gemini_provider._model_name == "text-embedding-004"
        assert gemini_provider.get_vector_name() == "gemini-text-embedding-004"

    async def test_model_name_normalization(self, mock_genai_client):
        """Test stripping of models/ prefix."""
        provider = GeminiEmbeddingProvider(model_name="models/text-embedding-004", api_key="fake-key")
        assert provider._model_name == "text-embedding-004"

    async def test_embed_documents(self, gemini_provider, mock_genai_client):
        """Test batch embedding with mocked response."""
        documents = ["Hello world", "Test document"]
        embeddings = await gemini_provider.embed_documents(documents)
        
        assert len(embeddings) == len(documents)
        assert all(len(emb) == 768 for emb in embeddings)
        
        mock_genai_client.aio.models.embed_content.assert_called_once()
        args, kwargs = mock_genai_client.aio.models.embed_content.call_args
        assert kwargs['model'] == "text-embedding-004"
        assert kwargs['contents'] == documents

    async def test_embed_query(self, gemini_provider, mock_genai_client):
        """Test single query embedding."""
        query = "What is MCP?"
        embedding = await gemini_provider.embed_query(query)
        
        assert len(embedding) == 768
        mock_genai_client.aio.models.embed_content.assert_called_once()
        args, kwargs = mock_genai_client.aio.models.embed_content.call_args
        assert kwargs['contents'] == [query]

    async def test_get_vector_name(self, gemini_provider):
        """Test naming convention gemini-{slug}."""
        assert gemini_provider.get_vector_name() == "gemini-text-embedding-004"

    async def test_get_vector_size_cached(self, gemini_provider):
        """Test dimension when cached."""
        gemini_provider._dimension = 1536
        assert gemini_provider.get_vector_size() == 1536

    async def test_get_vector_size_probe(self, gemini_provider, mock_genai_client):
        """Test dimension probe when not cached."""
        assert gemini_provider._dimension is None
        size = gemini_provider.get_vector_size()
        
        assert size == 768
        assert gemini_provider._dimension == 768
        mock_genai_client.models.embed_content.assert_called_once()

    async def test_dimension_caching(self, gemini_provider, mock_genai_client):
        """Verify dimension is cached after first call."""
        assert gemini_provider._dimension is None
        
        # First call (async)
        await gemini_provider.embed_query("test")
        assert gemini_provider._dimension == 768
        
        # Second call (sync size check) - should use cache
        mock_genai_client.models.embed_content.reset_mock()
        assert gemini_provider.get_vector_size() == 768
        mock_genai_client.models.embed_content.assert_not_called()

@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not set")
class TestGeminiEmbeddingProviderIntegration:
    """Integration tests for GeminiEmbeddingProvider with real API."""

    @pytest.fixture
    def api_key(self):
        """Get API key from environment."""
        key = os.getenv("GOOGLE_API_KEY")
        if not key:
            pytest.skip("GOOGLE_API_KEY not set")
        return key

    @pytest.fixture
    def real_gemini_provider(self, api_key):
        """Real provider for integration tests."""
        return GeminiEmbeddingProvider(model_name="text-embedding-004", api_key=api_key)

    async def test_real_embed_documents(self, real_gemini_provider):
        """Test with real API."""
        documents = ["This is a test document.", "This is another test document."]
        embeddings = await real_gemini_provider.embed_documents(documents)
        
        assert len(embeddings) == len(documents)
        assert all(len(emb) > 0 for emb in embeddings)
        
        embedding1 = np.array(embeddings[0])
        embedding2 = np.array(embeddings[1])
        assert not np.array_equal(embedding1, embedding2)

    async def test_real_embed_query(self, real_gemini_provider):
        """Test query embedding."""
        query = "This is a test query."
        embedding = await real_gemini_provider.embed_query(query)
        
        assert len(embedding) > 0
        
        # Consistency check
        embedding2 = await real_gemini_provider.embed_query(query)
        np.testing.assert_array_almost_equal(np.array(embedding), np.array(embedding2))

    async def test_real_consistency(self, real_gemini_provider):
        """Verify same input produces same output."""
        text = "Consistency test"
        res1 = await real_gemini_provider.embed_query(text)
        res2 = await real_gemini_provider.embed_query(text)
        np.testing.assert_array_almost_equal(np.array(res1), np.array(res2))

    async def test_real_vector_metadata(self, real_gemini_provider):
        """Test vector name and size."""
        assert real_gemini_provider.get_vector_name().startswith("gemini-")
        size = real_gemini_provider.get_vector_size()
        assert size > 0
        
        # Verify it matches embedding length
        embedding = await real_gemini_provider.embed_query("test")
        assert len(embedding) == size
