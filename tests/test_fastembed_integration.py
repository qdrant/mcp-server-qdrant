import numpy as np
import pytest
from fastembed import TextEmbedding

from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider


@pytest.mark.asyncio
class TestFastEmbedProviderIntegration:
    """Integration tests for FastEmbedProvider."""

    async def test_initialization(self):
        """Test that the provider can be initialized with a valid model."""
        provider = FastEmbedProvider("sentence-transformers/all-MiniLM-L6-v2")
        assert provider.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert isinstance(provider.embedding_model, TextEmbedding)

    async def test_embed_documents(self):
        """Test that documents can be embedded."""
        provider = FastEmbedProvider("sentence-transformers/all-MiniLM-L6-v2")
        documents = ["This is a test document.", "This is another test document."]

        embeddings = await provider.embed_documents(documents)

        # Check that we got the right number of embeddings
        assert len(embeddings) == len(documents)

        # Check that embeddings have the expected shape
        # The exact dimension depends on the model, but should be consistent
        assert len(embeddings[0]) > 0
        assert all(len(embedding) == len(embeddings[0]) for embedding in embeddings)

        # Check that embeddings are different for different documents
        # Convert to numpy arrays for easier comparison
        embedding1 = np.array(embeddings[0])
        embedding2 = np.array(embeddings[1])
        assert not np.array_equal(embedding1, embedding2)

    async def test_embed_query(self):
        """Test that queries can be embedded."""
        provider = FastEmbedProvider("sentence-transformers/all-MiniLM-L6-v2")
        query = "This is a test query."

        embedding = await provider.embed_query(query)

        # Check that embedding has the expected shape
        assert len(embedding) > 0

        # Embed the same query again to check consistency
        embedding2 = await provider.embed_query(query)
        assert len(embedding) == len(embedding2)

        # The embeddings should be identical for the same input
        np.testing.assert_array_almost_equal(np.array(embedding), np.array(embedding2))

    async def test_get_vector_name(self):
        """Test that the vector name is generated correctly."""
        provider = FastEmbedProvider("sentence-transformers/all-MiniLM-L6-v2")
        vector_name = provider.get_vector_name()

        # Check that the vector name follows the expected format
        assert vector_name.startswith("fast-")
        assert "minilm" in vector_name.lower()
