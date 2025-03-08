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
from typing import Dict, List, Optional, Set

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
        global_memory_id = await connector.store_memory("This is a global memory", None)
        print(f"Stored global memory with ID: {global_memory_id}")
        
        cat_memory_id1 = await connector.store_memory("This is a memory about cats", "cats")
        print(f"Stored cat memory 1 with ID: {cat_memory_id1}")
        
        cat_memory_id2 = await connector.store_memory("This is another memory about cats", "cats")
        print(f"Stored cat memory 2 with ID: {cat_memory_id2}")
        
        dog_memory_id = await connector.store_memory("This is a memory about dogs", "dogs")
        print(f"Stored dog memory with ID: {dog_memory_id}")
        
        # Test listing collections
        collections = await connector.list_collections()
        print(f"Available collections: {collections}")
        assert "cats" in collections, "Collection 'cats' not found"
        assert "dogs" in collections, "Collection 'dogs' not found"
        
        # Test finding memories in specific collections
        cat_memories = await connector.find_memories("cats", "cats")
        print(f"Cat memories: {json.dumps(cat_memories, indent=2)}")
        assert len(cat_memories) > 0, "No cat memories found"
        assert "id" in cat_memories[0], "Memory ID not found in result"
        assert "content" in cat_memories[0], "Memory content not found in result"
        
        dog_memories = await connector.find_memories("dogs", "dogs")
        print(f"Dog memories: {json.dumps(dog_memories, indent=2)}")
        assert len(dog_memories) > 0, "No dog memories found"
        assert "id" in dog_memories[0], "Memory ID not found in result"
        assert "content" in dog_memories[0], "Memory content not found in result"
        
        # Test finding memories across collections
        all_memories = await connector.find_memories_across_collections("pets")
        print(f"All memories: {json.dumps(all_memories, indent=2)}")
        
        # Test replacing memories
        updated_cat_memory_id = await connector.store_memory(
            "This is an updated memory about cats",
            "cats",
            replace_memory_ids=[cat_memory_id1]
        )
        print(f"Updated cat memory with ID: {updated_cat_memory_id}")
        
        # Verify the memory was replaced
        cat_memories_after_update = await connector.find_memories("cats", "cats")
        print(f"Cat memories after update: {json.dumps(cat_memories_after_update, indent=2)}")
        
        # Check if the old memory ID is no longer present
        cat_memory_ids = [memory["id"] for memory in cat_memories_after_update]
        assert cat_memory_id1 not in cat_memory_ids, "Old memory ID should not be present after replacement"
        
        # Test deleting memories
        deleted_count = await connector.delete_memories([cat_memory_id2, dog_memory_id])
        print(f"Deleted {deleted_count} memories")
        assert deleted_count == 2, f"Expected to delete 2 memories, but deleted {deleted_count}"
        
        # Verify memories were deleted
        cat_memories_after_delete = await connector.find_memories("cats", "cats")
        print(f"Cat memories after delete: {json.dumps(cat_memories_after_delete, indent=2)}")
        assert len(cat_memories_after_delete) < len(cat_memories_after_update), "Cat memory not deleted"
        
        dog_memories_after_delete = await connector.find_memories("dogs", "dogs")
        print(f"Dog memories after delete: {json.dumps(dog_memories_after_delete, indent=2)}")
        assert len(dog_memories_after_delete) == 0, "Dog memory not deleted"
        
        # Test invalid collection name
        try:
            await connector.store_memory("This should fail", "invalid name with spaces")
            assert False, "Should have failed with invalid collection name"
        except ValueError as e:
            print(f"Expected error for invalid collection name: {e}")
        
        print("All tests passed!")


async def test_protected_collections():
    """Test protected collections functionality."""
    print("\nTesting protected collections functionality...")
    
    # Create a temporary directory for Qdrant local storage
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create embedding provider
        provider = create_embedding_provider(
            provider_type="fastembed",
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Create QdrantConnector with multi-collection mode enabled and protected collections
        connector = QdrantConnector(
            qdrant_url=None,
            qdrant_api_key=None,
            collection_name="global",
            embedding_provider=provider,
            qdrant_local_path=temp_dir,
            multi_collection_mode=True,
            collection_prefix="test_",
            protected_collections={"cats"}  # Protect the "cats" collection
        )
        
        # Test storing memories in different collections
        global_memory_id = await connector.store_memory("This is a global memory", None)
        print(f"Stored global memory with ID: {global_memory_id}")
        
        # Store a memory in the unprotected "dogs" collection
        dog_memory_id = await connector.store_memory("This is a memory about dogs", "dogs")
        print(f"Stored dog memory with ID: {dog_memory_id}")
        
        # Test storing in a protected collection (should fail)
        try:
            await connector.store_memory("This should fail", "cats")
            assert False, "Should have failed to store in protected collection"
        except ValueError as e:
            print(f"Expected error for protected collection: {e}")
        
        # Create a memory in the protected collection by temporarily removing protection
        connector._protected_collections.remove("cats")
        cat_memory_id = await connector.store_memory("This is a memory about cats", "cats")
        print(f"Stored cat memory with ID: {cat_memory_id}")
        connector._protected_collections.add("cats")
        
        # Test finding memories in protected collection (should work)
        cat_memories = await connector.find_memories("cats", "cats")
        print(f"Cat memories: {json.dumps(cat_memories, indent=2)}")
        assert len(cat_memories) > 0, "No cat memories found"
        assert cat_memories[0].get("readonly", False), "Memory should be marked as readonly"
        
        # Test deleting from protected collection (should fail)
        try:
            await connector.delete_memories([cat_memory_id])
            assert False, "Should have failed to delete from protected collection"
        except ValueError as e:
            print(f"Expected error for deleting from protected collection: {e}")
        
        # Test deleting from unprotected collection (should work)
        deleted_count = await connector.delete_memories([dog_memory_id])
        print(f"Deleted {deleted_count} memories")
        assert deleted_count == 1, f"Expected to delete 1 memory, but deleted {deleted_count}"
        
        # Test finding memories across collections
        all_memories = await connector.find_memories_across_collections("pets")
        print(f"All memories: {json.dumps(all_memories, indent=2)}")
        
        # Verify readonly status in results
        if "cats" in all_memories:
            for memory in all_memories["cats"]:
                assert memory.get("readonly", False), "Cat memory should be marked as readonly"
        
        print("All protected collections tests passed!")


async def test_protect_all_collections():
    """Test protecting all collections functionality."""
    print("\nTesting protect all collections functionality...")
    
    # Create a temporary directory for Qdrant local storage
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create embedding provider
        provider = create_embedding_provider(
            provider_type="fastembed",
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Create QdrantConnector with multi-collection mode enabled and all collections protected
        connector = QdrantConnector(
            qdrant_url=None,
            qdrant_api_key=None,
            collection_name="global",
            embedding_provider=provider,
            qdrant_local_path=temp_dir,
            multi_collection_mode=True,
            collection_prefix="test_",
            protected_collections={"*"}  # Protect all collections
        )
        
        # Create some collections first by temporarily removing protection
        connector._protected_collections.clear()
        
        # Store memories in different collections
        global_memory_id = await connector.store_memory("This is a global memory", None)
        print(f"Stored global memory with ID: {global_memory_id}")
        
        cat_memory_id = await connector.store_memory("This is a memory about cats", "cats")
        print(f"Stored cat memory with ID: {cat_memory_id}")
        
        dog_memory_id = await connector.store_memory("This is a memory about dogs", "dogs")
        print(f"Stored dog memory with ID: {dog_memory_id}")
        
        # Re-enable protection for all collections
        connector._protected_collections = {"*"}
        
        # Test storing in any collection (should fail)
        try:
            await connector.store_memory("This should fail", "cats")
            assert False, "Should have failed to store in protected collection"
        except ValueError as e:
            print(f"Expected error for protected collection: {e}")
            
        try:
            await connector.store_memory("This should fail", "dogs")
            assert False, "Should have failed to store in protected collection"
        except ValueError as e:
            print(f"Expected error for protected collection: {e}")
            
        try:
            await connector.store_memory("This should fail", "global")
            assert False, "Should have failed to store in protected collection"
        except ValueError as e:
            print(f"Expected error for protected collection: {e}")
        
        # Test finding memories in protected collections (should work)
        cat_memories = await connector.find_memories("cats", "cats")
        print(f"Cat memories: {json.dumps(cat_memories, indent=2)}")
        assert len(cat_memories) > 0, "No cat memories found"
        assert cat_memories[0].get("readonly", False), "Memory should be marked as readonly"
        
        dog_memories = await connector.find_memories("dogs", "dogs")
        print(f"Dog memories: {json.dumps(dog_memories, indent=2)}")
        assert len(dog_memories) > 0, "No dog memories found"
        assert dog_memories[0].get("readonly", False), "Memory should be marked as readonly"
        
        # Test deleting from any collection (should fail)
        try:
            await connector.delete_memories([cat_memory_id])
            assert False, "Should have failed to delete from protected collection"
        except ValueError as e:
            print(f"Expected error for deleting from protected collection: {e}")
            
        try:
            await connector.delete_memories([dog_memory_id])
            assert False, "Should have failed to delete from protected collection"
        except ValueError as e:
            print(f"Expected error for deleting from protected collection: {e}")
        
        # Test finding memories across collections
        all_memories = await connector.find_memories_across_collections("pets")
        print(f"All memories: {json.dumps(all_memories, indent=2)}")
        
        # Verify readonly status in results for all collections
        for collection, memories in all_memories.items():
            for memory in memories:
                assert memory.get("readonly", False), f"{collection} memory should be marked as readonly"
        
        print("All protect all collections tests passed!")


if __name__ == "__main__":
    asyncio.run(test_multi_collection())
    asyncio.run(test_protected_collections())
    asyncio.run(test_protect_all_collections()) 