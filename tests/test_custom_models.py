"""Test custom model functionality with FastEmbed."""

import os
from unittest.mock import patch
from mcp_server_qdrant.settings import EmbeddingProviderSettings, CustomModelSettings
from mcp_server_qdrant.embeddings.factory import create_embedding_provider
from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider


class TestCustomModelIntegration:
    """Test custom model integration."""

    def test_custom_model_settings_parsing(self):
        """Test that custom model settings are parsed correctly."""
        env_vars = {
            "EMBEDDING_USE_CUSTOM_MODEL": "true",
            "EMBEDDING_CUSTOM_MODEL_NAME": "test-model",
            "EMBEDDING_CUSTOM_HF_MODEL_ID": "test/model",
            "EMBEDDING_CUSTOM_VECTOR_DIMENSION": "384",
            "EMBEDDING_CUSTOM_POOLING_TYPE": "CLS",
            "EMBEDDING_CUSTOM_NORMALIZATION": "false",
            "EMBEDDING_CUSTOM_MODEL_FILE": "onnx/model.onnx",
            "EMBEDDING_CUSTOM_ADDITIONAL_FILES": "tokenizer.json,config.json",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            settings = EmbeddingProviderSettings()
            assert settings.use_custom_model is True
            assert settings.custom_model_name == "test-model"
            assert settings.custom_hf_model_id == "test/model"
            assert settings.custom_vector_dimension == 384
            assert settings.custom_pooling_type == "CLS"
            assert settings.custom_normalization is False
            assert settings.custom_model_file == "onnx/model.onnx"
            assert settings.custom_additional_files == "tokenizer.json,config.json"

            custom_settings = settings.get_custom_model_settings()
            assert custom_settings.additional_files == ["tokenizer.json", "config.json"]

    def test_custom_model_settings_defaults(self):
        """Test that custom model settings have correct defaults."""
        env_vars = {
            "EMBEDDING_USE_CUSTOM_MODEL": "false",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            settings = EmbeddingProviderSettings()
            assert settings.use_custom_model is False
            assert settings.custom_model_name is None
            assert settings.custom_hf_model_id is None
            assert settings.custom_vector_dimension is None
            assert settings.custom_pooling_type == "MEAN"
            assert settings.custom_normalization is True
            assert settings.custom_model_file == "onnx/model.onnx"
            assert settings.custom_model_url is None
            assert settings.custom_additional_files is None

    @patch("mcp_server_qdrant.embeddings.fastembed.TextEmbedding")
    @patch.object(FastEmbedProvider, "_register_custom_model")
    def test_custom_model_provider_creation(self, mock_register, mock_text_embedding):
        """Test that custom model provider is created correctly."""
        env_vars = {
            "EMBEDDING_USE_CUSTOM_MODEL": "true",
            "EMBEDDING_CUSTOM_MODEL_NAME": "test-model",
            "EMBEDDING_CUSTOM_HF_MODEL_ID": "test/model",
            "EMBEDDING_CUSTOM_VECTOR_DIMENSION": "384",
            "EMBEDDING_CUSTOM_POOLING_TYPE": "CLS",
            "EMBEDDING_CUSTOM_NORMALIZATION": "false",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            settings = EmbeddingProviderSettings()
            expected = CustomModelSettings(
                model_name="test-model",
                hf_model_id="test/model",
                model_url=None,
                pooling_type="CLS",
                normalization=False,
                vector_dimension=384,
                model_file="onnx/model.onnx",
                additional_files=None,
            )
            _ = create_embedding_provider(settings)
            mock_register.assert_called_once_with(expected)
            mock_text_embedding.assert_called_once_with("test-model")

    @patch("mcp_server_qdrant.embeddings.fastembed.TextEmbedding")
    @patch.object(FastEmbedProvider, "_register_custom_model")
    def test_custom_model_with_additional_files(self, mock_register, mock_text_embedding):
        """Test custom model creation with additional files."""
        env_vars = {
            "EMBEDDING_USE_CUSTOM_MODEL": "true",
            "EMBEDDING_CUSTOM_MODEL_NAME": "test-model",
            "EMBEDDING_CUSTOM_HF_MODEL_ID": "test/model",
            "EMBEDDING_CUSTOM_VECTOR_DIMENSION": "384",
            "EMBEDDING_CUSTOM_ADDITIONAL_FILES": "tokenizer.json,config.json,special_tokens_map.json",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            settings = EmbeddingProviderSettings()
            expected = CustomModelSettings(
                model_name="test-model",
                hf_model_id="test/model",
                model_url=None,
                pooling_type="MEAN",
                normalization=True,
                vector_dimension=384,
                model_file="onnx/model.onnx",
                additional_files=[
                    "tokenizer.json",
                    "config.json",
                    "special_tokens_map.json"
                ],
            )
            _ = create_embedding_provider(settings)
            mock_register.assert_called_once_with(expected)

    @patch("mcp_server_qdrant.embeddings.fastembed.TextEmbedding")
    @patch.object(FastEmbedProvider, "_register_custom_model")
    def test_custom_model_with_url(self, mock_register, mock_text_embedding):
        """Test custom model creation with direct URL."""
        env_vars = {
            "EMBEDDING_USE_CUSTOM_MODEL": "true",
            "EMBEDDING_CUSTOM_MODEL_NAME": "test-model",
            "EMBEDDING_CUSTOM_MODEL_URL": "https://example.com/model.tar.gz",
            "EMBEDDING_CUSTOM_VECTOR_DIMENSION": "768",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            settings = EmbeddingProviderSettings()
            expected = CustomModelSettings(
                model_name="test-model",
                hf_model_id=None,
                model_url="https://example.com/model.tar.gz",
                pooling_type="MEAN",
                normalization=True,
                vector_dimension=768,
                model_file="onnx/model.onnx",
                additional_files=None,
            )
            _ = create_embedding_provider(settings)
            mock_register.assert_called_once_with(expected)

    def test_standard_model_not_affected(self):
        """Test that standard model creation is not affected when custom model is disabled."""
        env_vars = {
            "EMBEDDING_USE_CUSTOM_MODEL": "false",
            "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            settings = EmbeddingProviderSettings()
            assert settings.use_custom_model is False
            assert settings.model_name == "sentence-transformers/all-MiniLM-L6-v2"
