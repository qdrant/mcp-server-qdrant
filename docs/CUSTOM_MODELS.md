# Custom Model Configuration

This document explains how to configure and use custom embedding models with the MCP Qdrant server using FastEmbed's `add_custom_model` functionality.

## Overview

The MCP Qdrant server supports custom embedding models through FastEmbed's custom model feature. This allows you to use your own fine-tuned models or models not included in FastEmbed's default registry.

## Configuration

Custom models are configured through environment variables. Here's how to set them up:

### Basic Custom Model Setup

1. Set `EMBEDDING_USE_CUSTOM_MODEL=true` to enable custom model mode
2. Configure the required custom model parameters

### Required Environment Variables

When using custom models, the following variables are required:

- `EMBEDDING_CUSTOM_MODEL_NAME`: A unique name for your custom model
- `EMBEDDING_CUSTOM_HF_MODEL_ID`: HuggingFace model ID (e.g., `intfloat/multilingual-e5-small`)
- `EMBEDDING_CUSTOM_VECTOR_DIMENSION`: The output dimension of your model

### Optional Environment Variables

- `EMBEDDING_CUSTOM_POOLING_TYPE`: Pooling strategy (`MEAN`, `CLS`, `MAX`, etc.) - defaults to `MEAN`
- `EMBEDDING_CUSTOM_NORMALIZATION`: Whether to normalize embeddings (`true`/`false`) - defaults to `true`
- `EMBEDDING_CUSTOM_MODEL_FILE`: Specific model file to use (e.g., `onnx/model.onnx`)
- `EMBEDDING_CUSTOM_MODEL_URL`: Direct URL to download the model (alternative to HuggingFace)
- `EMBEDDING_CUSTOM_ADDITIONAL_FILES`: Comma-separated list of additional files to download (e.g., `tokenizer.json,config.json`)

## Examples

### Example 1: Multilingual E5 Model

```bash
EMBEDDING_USE_CUSTOM_MODEL=true
EMBEDDING_CUSTOM_MODEL_NAME=my-e5-model
EMBEDDING_CUSTOM_HF_MODEL_ID=intfloat/multilingual-e5-large-instruct
EMBEDDING_CUSTOM_VECTOR_DIMENSION=1024
EMBEDDING_CUSTOM_POOLING_TYPE=MEAN
EMBEDDING_CUSTOM_NORMALIZATION=true
# EMBEDDING_CUSTOM_ADDITIONAL_FILES=tokenizer.json,vocab.txt
```

## Model Requirements

Your custom model should:

1. Be compatible with FastEmbed's ONNX runtime
2. Have a known output dimension
3. Be accessible either through HuggingFace Hub or a direct URL

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure the HuggingFace model ID is correct and the model exists
2. **Dimension mismatch**: Verify that `EMBEDDING_CUSTOM_VECTOR_DIMENSION` matches your model's actual output
3. **Download failures**: Check network connectivity and model availability
4. **Additional files not found**: Ensure all files listed in `EMBEDDING_CUSTOM_ADDITIONAL_FILES` exist in the model repository

### Validation

The server will validate your custom model configuration on startup and provide helpful error messages if something is misconfigured.

## Performance Considerations

- Custom models are downloaded and cached on first use
- Ensure you have sufficient disk space for model files
- Consider the model size and inference speed for your use case
- Additional files increase download time and storage requirements

## Security Notes

- Be cautious when using models from untrusted sources
- Validate model URLs and ensure they point to legitimate model repositories
- Consider using private model repositories for sensitive applications