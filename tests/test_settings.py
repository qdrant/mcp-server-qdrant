import os
from unittest.mock import patch

from mcp_server_qdrant.embeddings.types import EmbeddingProviderType
from mcp_server_qdrant.settings import (
    DEFAULT_TOOL_FIND_DESCRIPTION,
    DEFAULT_TOOL_STORE_DESCRIPTION,
    EmbeddingProviderSettings,
    QdrantSettings,
    ToolSettings,
)


class TestQdrantSettings:
    def test_default_values(self):
        """Test that required fields raise errors when not provided."""

        # Should not raise error because there are no required fields
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

    def test_custom_values(self):
        """Test custom values for embedding provider settings."""
        import os

        # Set environment variables
        os.environ["EMBEDDING_PROVIDER"] = "google_genai"
        os.environ["EMBEDDING_MODEL"] = "text-embedding-004"
        os.environ["GOOGLE_API_KEY"] = "test-api-key"
        os.environ["GOOGLE_CLOUD_PROJECT"] = "test-project"
        os.environ["GOOGLE_CLOUD_LOCATION"] = "us-west1"
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"

        try:
            settings = EmbeddingProviderSettings()
            assert settings.provider_type == EmbeddingProviderType.GOOGLE_GENAI
            assert settings.model_name == "text-embedding-004"
            assert settings.api_key == "test-api-key"
            assert settings.project == "test-project"
            assert settings.location == "us-west1"
            assert settings.use_vertex_ai is True
        finally:
            # Clean up environment variables
            for key in [
                "EMBEDDING_PROVIDER",
                "EMBEDDING_MODEL",
                "GOOGLE_API_KEY",
                "GOOGLE_CLOUD_PROJECT",
                "GOOGLE_CLOUD_LOCATION",
                "GOOGLE_GENAI_USE_VERTEXAI",
            ]:
                os.environ.pop(key, None)

    def test_string_to_enum_conversion(self):
        """Test that string values are properly converted to enum."""
        import os

        # Test with string value that should convert to enum
        os.environ["EMBEDDING_PROVIDER"] = "google_genai"

        try:
            settings = EmbeddingProviderSettings()
            assert settings.provider_type == EmbeddingProviderType.GOOGLE_GENAI
            assert isinstance(settings.provider_type, EmbeddingProviderType)
        finally:
            os.environ.pop("EMBEDDING_PROVIDER", None)


class TestToolSettings:
    def test_default_values(self):
        """Test that default values are set correctly when no env vars are provided."""
        settings = ToolSettings()
        assert settings.tool_store_description == DEFAULT_TOOL_STORE_DESCRIPTION
        assert settings.tool_find_description == DEFAULT_TOOL_FIND_DESCRIPTION

    @patch.dict(
        os.environ,
        {"TOOL_STORE_DESCRIPTION": "Custom store description"},
    )
    def test_custom_store_description(self):
        """Test loading custom store description from environment variable."""
        settings = ToolSettings()
        assert settings.tool_store_description == "Custom store description"
        assert settings.tool_find_description == DEFAULT_TOOL_FIND_DESCRIPTION

    @patch.dict(
        os.environ,
        {"TOOL_FIND_DESCRIPTION": "Custom find description"},
    )
    def test_custom_find_description(self):
        """Test loading custom find description from environment variable."""
        settings = ToolSettings()
        assert settings.tool_store_description == DEFAULT_TOOL_STORE_DESCRIPTION
        assert settings.tool_find_description == "Custom find description"

    @patch.dict(
        os.environ,
        {
            "TOOL_STORE_DESCRIPTION": "Custom store description",
            "TOOL_FIND_DESCRIPTION": "Custom find description",
        },
    )
    def test_all_custom_values(self):
        """Test loading all custom values from environment variables."""
        settings = ToolSettings()
        assert settings.tool_store_description == "Custom store description"
        assert settings.tool_find_description == "Custom find description"
