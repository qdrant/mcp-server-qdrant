#!/usr/bin/env python
"""Test script to verify unnamed vectors support in the MCP server"""

import asyncio
import os
import sys

# Set environment variables
os.environ["USE_UNNAMED_VECTORS"] = "true"
os.environ["EMBEDDING_MODEL"] = "sentence-transformers/all-MiniLM-L6-v2"

# Add src to path
sys.path.insert(0, "src")

from mcp_server_qdrant.embeddings.factory import create_embedding_provider
from mcp_server_qdrant.settings import EmbeddingProviderSettings


async def test_unnamed_vectors():
    print("Testing unnamed vectors support...")
    
    # Create settings
    settings = EmbeddingProviderSettings()
    print(f"Settings loaded:")
    print(f"  - Provider type: {settings.provider_type}")
    print(f"  - Model name: {settings.model_name}")
    print(f"  - Use unnamed vectors: {settings.use_unnamed_vectors}")
    
    # Create embedding provider
    provider = create_embedding_provider(settings)
    print(f"\nEmbedding provider created: {type(provider).__name__}")
    print(f"Vector name: {provider.get_vector_name()}")
    print(f"Vector size: {provider.get_vector_size()}")
    
    # Test embedding
    test_text = "This is a test document for unnamed vectors"
    print(f"\nEmbedding test text: '{test_text}'")
    
    embeddings = await provider.embed_documents([test_text])
    print(f"Embedding generated, dimension: {len(embeddings[0])}")
    
    # Check if it's using unnamed vectors
    if provider.get_vector_name() == "unnamed_vector":
        print("\n✅ SUCCESS: Unnamed vectors are correctly configured!")
    else:
        print(f"\n❌ ERROR: Expected 'unnamed_vector' but got '{provider.get_vector_name()}'")
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(test_unnamed_vectors())
    sys.exit(0 if success else 1)