"""Test custom model functionality with FastEmbed."""

import os
import pytest
from unittest.mock import patch, MagicMock

from mcp_server_qdrant.settings import EmbeddingProviderSettings
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

    def test_custom_model_validation_missing_name(self):
        """Test that validation fails when custom model name is missing."""
        env_vars = {
            "EMBEDDING_USE_CUSTOM_MODEL": "true",
            "EMBEDDING_CUSTOM_HF_MODEL_ID": "test/model",
            "EMBEDDING_CUSTOM_VECTOR_DIMENSION": "384",
        }
        
        with patch.dict(os.environ, env_vars, clear=False):
            settings = EmbeddingProviderSettings()
            with pytest.raises(ValueError, match="custom_model_name is required"):
                settings.get_custom_model_settings()

    def test_custom_model_validation_missing_dimension(self):
        """Test that validation fails when vector dimension is missing."""
        env_vars = {
            "EMBEDDING_USE_CUSTOM_MODEL": "true",
            "EMBEDDING_CUSTOM_MODEL_NAME": "test-model",
            "EMBEDDING_CUSTOM_HF_MODEL_ID": "test/model",
        }
        
        with patch.dict(os.environ, env_vars, clear=False):
            settings = EmbeddingProviderSettings()
            with pytest.raises(ValueError, match="custom_vector_dimension is required"):
                settings.get_custom_model_settings()

    @patch('mcp_server_qdrant.embeddings.fastembed.FastEmbedding')
    @patch('mcp_server_qdrant.embeddings.fastembed.add_custom_model')
    def test_custom_model_provider_creation(self, mock_add_custom_model, mock_fastembed):
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
            provider = create_embedding_provider(settings)
            
            # Verify add_custom_model was called
            mock_add_custom_model.assert_called_once_with(
                model_name="test-model",
                hf_model_id="test/model",
                pooling_type="CLS",
                normalize=False,
                model_file="onnx/model.onnx",
                model_url=None
            )
            
            # Verify FastEmbedding was initialized with custom model
            mock_fastembed.assert_called_once_with(
                model_name="test-model",
                cache_dir=None
            )

    @patch('mcp_server_qdrant.embeddings.fastembed.FastEmbedding')
    @patch('mcp_server_qdrant.embeddings.fastembed.add_custom_model')
    def test_custom_model_with_url(self, mock_add_custom_model, mock_fastembed):
        """Test custom model creation with direct URL."""
        env_vars = {
            "EMBEDDING_USE_CUSTOM_MODEL": "true",
            "EMBEDDING_CUSTOM_MODEL_NAME": "test-model",
            "EMBEDDING_CUSTOM_MODEL_URL": "https://example.com/model.tar.gz",
            "EMBEDDING_CUSTOM_VECTOR_DIMENSION": "768",
        }
        
        with patch.dict(os.environ, env_vars, clear=False):
            settings = EmbeddingProviderSettings()
            provider = create_embedding_provider(settings)
            
            # Verify add_custom_model was called with URL
            mock_add_custom_model.assert_called_once_with(
                model_name="test-model",
                hf_model_id=None,
                pooling_type="MEAN",
                normalize=True,
                model_file="onnx/model.onnx",
                model_url="https://example.com/model.tar.gz"
            )

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
