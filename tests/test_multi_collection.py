#!/usr/bin/env python3
"""
Test script for multi-collection functionality in mcp-server-qdrant.
This script tests the basic operations of storing and retrieving memories
across multiple collections.
"""

import asyncio
import json
import os
import tempfile
import uuid
from typing import Dict, List, Optional

from mcp_server_qdrant.embeddings.factory import create_embedding_provider
from mcp_server_qdrant.qdrant import QdrantConnector


async def test_multi_collection():
    """Test multi-collection functionality."""
    print("Testing multi-collection functionality...")
    
    # Create a temporary directory for Qdrant local storage
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create embedding provider
        provider = create_embedding_provider(
            provider_type="fastembed",
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Create QdrantConnector with multi-collection mode enabled
        connector = QdrantConnector(
            qdrant_url=None,
            qdrant_api_key=None,
            collection_name="global",
            embedding_provider=provider,
            qdrant_local_path=temp_dir,
            multi_collection_mode=True,
            collection_prefix="test_",
        )
        
        # Test storing memories in different collections
        await connector.store_memory("This is a global memory", None)
        await connector.store_memory("This is a memory about cats", "cats")
        await connector.store_memory("This is another memory about cats", "cats")
        await connector.store_memory("This is a memory about dogs", "dogs")
        
        # Test listing collections
        collections = await connector.list_collections()
        print(f"Available collections: {collections}")
        assert "cats" in collections, "Collection 'cats' not found"
        assert "dogs" in collections, "Collection 'dogs' not found"
        
        # Test finding memories in specific collections
        cat_memories = await connector.find_memories("cats", "cats")
        print(f"Cat memories: {cat_memories}")
        assert len(cat_memories) > 0, "No cat memories found"
        
        dog_memories = await connector.find_memories("dogs", "dogs")
        print(f"Dog memories: {dog_memories}")
        assert len(dog_memories) > 0, "No dog memories found"
        
        # Test finding memories across collections
        all_memories = await connector.find_memories_across_collections("pets")
        print(f"All memories: {json.dumps(all_memories, indent=2)}")
        
        # Test invalid collection name
        try:
            await connector.store_memory("This should fail", "invalid name with spaces")
            assert False, "Should have failed with invalid collection name"
        except ValueError as e:
            print(f"Expected error for invalid collection name: {e}")
        
        print("All tests passed!")


if __name__ == "__main__":
    asyncio.run(test_multi_collection()) 