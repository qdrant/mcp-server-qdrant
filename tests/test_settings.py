import itertools
import typing

import pytest
from pydantic import ValidationError

from mcp_server_qdrant.common.filters import make_filter
from mcp_server_qdrant.embeddings.types import EmbeddingProviderType
from mcp_server_qdrant.settings import (
    DEFAULT_TOOL_FIND_DESCRIPTION,
    DEFAULT_TOOL_STORE_DESCRIPTION,
    SUPPORTED_CONDITIONS,
    EmbeddingProviderSettings,
    FilterableField,
    QdrantSettings,
    ToolSettings,
)


class TestQdrantSettings:
    def test_default_values(self):
        """Test that required fields raise errors when not provided."""

        # Should not raise error because there are no required fields
        QdrantSettings()

    def test_minimal_config(self, monkeypatch):
        """Test loading minimal configuration from environment variables."""
        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("COLLECTION_NAME", "test_collection")

        settings = QdrantSettings()
        assert settings.location == "http://localhost:6333"
        assert settings.collection_name == "test_collection"
        assert settings.api_key is None
        assert settings.local_path is None

    def test_full_config(self, monkeypatch):
        """Test loading full configuration from environment variables."""
        monkeypatch.setenv("QDRANT_URL", "http://qdrant.example.com:6333")
        monkeypatch.setenv("QDRANT_API_KEY", "test_api_key")
        monkeypatch.setenv("COLLECTION_NAME", "my_memories")
        monkeypatch.setenv("QDRANT_SEARCH_LIMIT", "15")
        monkeypatch.setenv("QDRANT_READ_ONLY", "1")

        settings = QdrantSettings()
        assert settings.location == "http://qdrant.example.com:6333"
        assert settings.api_key == "test_api_key"
        assert settings.collection_name == "my_memories"
        assert settings.search_limit == 15
        assert settings.read_only is True

    def test_local_path_config(self, monkeypatch):
        """Test loading local path configuration from environment variables."""
        monkeypatch.setenv("QDRANT_LOCAL_PATH", "/path/to/local/qdrant")

        settings = QdrantSettings()
        assert settings.local_path == "/path/to/local/qdrant"

    def test_local_path_is_exclusive_with_url(self, monkeypatch):
        """Test that local path cannot be set if Qdrant URL is provided."""
        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("QDRANT_LOCAL_PATH", "/path/to/local/qdrant")

        with pytest.raises(ValueError):
            QdrantSettings()

        monkeypatch.delenv("QDRANT_URL", raising=False)
        monkeypatch.setenv("QDRANT_API_KEY", "test_api_key")
        with pytest.raises(ValueError):
            QdrantSettings()


class TestEmbeddingProviderSettings:
    def test_default_values(self):
        """Test default values are set correctly."""
        settings = EmbeddingProviderSettings()
        assert settings.provider_type == EmbeddingProviderType.FASTEMBED
        assert settings.model_name == "sentence-transformers/all-MiniLM-L6-v2"

    def test_custom_values(self, monkeypatch):
        """Test loading custom values from environment variables."""
        monkeypatch.setenv("EMBEDDING_MODEL", "custom_model")
        settings = EmbeddingProviderSettings()
        assert settings.provider_type == EmbeddingProviderType.FASTEMBED
        assert settings.model_name == "custom_model"


class TestToolSettings:
    def test_default_values(self):
        """Test that default values are set correctly when no env vars are provided."""
        settings = ToolSettings()
        assert settings.tool_store_description == DEFAULT_TOOL_STORE_DESCRIPTION
        assert settings.tool_find_description == DEFAULT_TOOL_FIND_DESCRIPTION

    def test_custom_store_description(self, monkeypatch):
        """Test loading custom store description from environment variable."""
        monkeypatch.setenv("TOOL_STORE_DESCRIPTION", "Custom store description")
        settings = ToolSettings()
        assert settings.tool_store_description == "Custom store description"
        assert settings.tool_find_description == DEFAULT_TOOL_FIND_DESCRIPTION

    def test_custom_find_description(self, monkeypatch):
        """Test loading custom find description from environment variable."""
        monkeypatch.setenv("TOOL_FIND_DESCRIPTION", "Custom find description")
        settings = ToolSettings()
        assert settings.tool_store_description == DEFAULT_TOOL_STORE_DESCRIPTION
        assert settings.tool_find_description == "Custom find description"

    def test_all_custom_values(self, monkeypatch):
        """Test loading all custom values from environment variables."""
        monkeypatch.setenv("TOOL_STORE_DESCRIPTION", "Custom store description")
        monkeypatch.setenv("TOOL_FIND_DESCRIPTION", "Custom find description")
        settings = ToolSettings()
        assert settings.tool_store_description == "Custom store description"
        assert settings.tool_find_description == "Custom find description"


FIELD_TYPES = typing.get_args(FilterableField.model_fields["field_type"].annotation)
CONDITIONS = typing.get_args(
    typing.get_args(FilterableField.model_fields["condition"].annotation)[0]
)
ALL_PAIRS = list(itertools.product(FIELD_TYPES, CONDITIONS))
VALID_PAIRS = [(t, c) for t, c in ALL_PAIRS if c in SUPPORTED_CONDITIONS[t]]
INVALID_PAIRS = [(t, c) for t, c in ALL_PAIRS if c not in SUPPORTED_CONDITIONS[t]]
SAMPLE_VALUES = {"keyword": "a", "integer": 1, "float": 1.5, "boolean": True}


class TestFilterableField:
    @pytest.mark.parametrize("field_type, condition", VALID_PAIRS)
    def test_supported_condition_is_accepted(self, field_type, condition):
        FilterableField(
            name="f", description="d", field_type=field_type, condition=condition
        )

    @pytest.mark.parametrize("field_type, condition", INVALID_PAIRS)
    def test_unsupported_condition_is_rejected(self, field_type, condition):
        with pytest.raises(ValidationError, match="is not supported for"):
            FilterableField(
                name="f", description="d", field_type=field_type, condition=condition
            )

    @pytest.mark.parametrize("field_type", FIELD_TYPES)
    def test_no_condition_is_accepted_for_every_type(self, field_type):
        FilterableField(name="f", description="d", field_type=field_type)

    @pytest.mark.parametrize("field_type, condition", ALL_PAIRS)
    def test_table_matches_make_filter(self, field_type, condition):
        """SUPPORTED_CONDITIONS must list exactly what make_filter can build."""
        # model_construct skips validation, so make_filter sees invalid pairs too.
        field = FilterableField.model_construct(
            name="f", description="d", field_type=field_type, condition=condition
        )
        value = SAMPLE_VALUES[field_type]
        if condition in ("any", "except"):
            value = [value]
        if condition in SUPPORTED_CONDITIONS[field_type]:
            make_filter({"f": field}, {"f": value})
        else:
            with pytest.raises(ValueError):
                make_filter({"f": field}, {"f": value})

    def test_invalid_condition_fails_when_settings_load(self, monkeypatch):
        monkeypatch.setenv(
            "FILTERABLE_FIELDS",
            '[{"name": "lang", "description": "Language", '
            '"field_type": "keyword", "condition": ">"}]',
        )
        with pytest.raises(ValidationError) as error:
            QdrantSettings()
        assert (
            "Condition '>' is not supported for keyword field 'lang'. "
            "Supported conditions: ==, !=, any, except." in str(error.value)
        )
