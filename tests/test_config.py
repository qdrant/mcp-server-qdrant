import os
from unittest.mock import patch

import pytest

from mcp_server_qdrant.embeddings.types import EmbeddingProviderType
from mcp_server_qdrant.settings import EmbeddingProviderSettings, QdrantSettings


class TestQdrantSettings:
    def test_default_values(self):
        """Test that required fields raise errors when not provided."""
        with pytest.raises(ValueError):
            # Should raise error because required fields are missing
            QdrantSettings()

    @patch.dict(
        os.environ,
        {"QDRANT_URL": "http://localhost:6333", "COLLECTION_NAME": "test_collection"},
    )
    def test_minimal_config(self):
        """Test loading minimal configuration from environment variables."""
        settings = QdrantSettings()
        assert settings.location == "http://localhost:6333"
        assert settings.collection_name == "test_collection"
        assert settings.api_key is None
        assert settings.local_path is None

    @patch.dict(
        os.environ,
        {
            "QDRANT_URL": "http://qdrant.example.com:6333",
            "QDRANT_API_KEY": "test_api_key",
            "COLLECTION_NAME": "my_memories",
            "QDRANT_LOCAL_PATH": "/tmp/qdrant",
        },
    )
    def test_full_config(self):
        """Test loading full configuration from environment variables."""
        settings = QdrantSettings()
        assert settings.location == "http://qdrant.example.com:6333"
        assert settings.api_key == "test_api_key"
        assert settings.collection_name == "my_memories"
        assert settings.local_path == "/tmp/qdrant"


class TestEmbeddingProviderSettings:
    def test_default_values(self):
        """Test default values are set correctly."""
        settings = EmbeddingProviderSettings()
        assert settings.provider_type == EmbeddingProviderType.FASTEMBED
        assert settings.model_name == "sentence-transformers/all-MiniLM-L6-v2"

    @patch.dict(
        os.environ,
        {"EMBEDDING_MODEL": "custom_model"},
    )
    def test_custom_values(self):
        """Test loading custom values from environment variables."""
        settings = EmbeddingProviderSettings()
        assert settings.provider_type == EmbeddingProviderType.FASTEMBED
        assert settings.model_name == "custom_model"
