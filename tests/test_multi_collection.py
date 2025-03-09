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
            collection_name="default",
            embedding_provider=provider,
            qdrant_local_path=temp_dir,
            multi_collection_mode=True,
            collection_prefix="test_",
        )
        
        # Test storing memories in different collections
        default_memory_id = await connector.store_memory("This is a default memory", None)
        print(f"Stored default memory with ID: {default_memory_id}")
        
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
            collection_name="default",
            embedding_provider=provider,
            qdrant_local_path=temp_dir,
            multi_collection_mode=True,
            collection_prefix="test_",
            protected_collections={"cats"}  # Protect the "cats" collection from deletion
        )
        
        # Test storing memories in different collections
        default_memory_id = await connector.store_memory("This is a default memory", None)
        print(f"Stored default memory with ID: {default_memory_id}")
        
        # Store a memory in the unprotected "dogs" collection
        dog_memory_id = await connector.store_memory("This is a memory about dogs", "dogs")
        print(f"Stored dog memory with ID: {dog_memory_id}")
        
        # Test storing in a protected collection (should work now since collections are insert-only)
        cat_memory_id = await connector.store_memory("This is a memory about cats", "cats")
        print(f"Stored cat memory with ID: {cat_memory_id}")
        
        # Test finding memories in protected collection (should work)
        cat_memories = await connector.find_memories("cats", "cats")
        print(f"Cat memories: {json.dumps(cat_memories, indent=2)}")
        assert len(cat_memories) > 0, "No cat memories found"
        assert cat_memories[0].get("readonly", False), "Memory should be marked as readonly"
        
        # Test deleting from protected collection (should work)
        deleted_count = await connector.delete_memories([cat_memory_id])
        print(f"Deleted count (should be 0 since memory was skipped): {deleted_count}")
        assert deleted_count == 0, f"Expected to delete 0 memories, but got {deleted_count}"
        
        # Verify the memory still exists
        cat_memories_after = await connector.find_memories("cats", "cats")
        print(f"Cat memories after delete attempt: {json.dumps(cat_memories_after, indent=2)}")
        assert len(cat_memories_after) == len(cat_memories), "Cat memory should still exist"
        
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
            collection_name="default",
            embedding_provider=provider,
            qdrant_local_path=temp_dir,
            multi_collection_mode=True,
            collection_prefix="test_",
            protected_collections={"*"}  # Protect all collections from deletion
        )
        
        # Store memories in different collections (should work with the new insert-only model)
        default_memory_id = await connector.store_memory("This is a default memory", None)
        print(f"Stored default memory with ID: {default_memory_id}")
        
        cat_memory_id = await connector.store_memory("This is a memory about cats", "cats")
        print(f"Stored cat memory with ID: {cat_memory_id}")
        
        dog_memory_id = await connector.store_memory("This is a memory about dogs", "dogs")
        print(f"Stored dog memory with ID: {dog_memory_id}")
        
        # Test finding memories in protected collections (should work)
        cat_memories = await connector.find_memories("cats", "cats")
        print(f"Cat memories: {json.dumps(cat_memories, indent=2)}")
        assert len(cat_memories) > 0, "No cat memories found"
        assert cat_memories[0].get("readonly", False), "Memory should be marked as readonly in the connector"
        
        dog_memories = await connector.find_memories("dogs", "dogs")
        print(f"Dog memories: {json.dumps(dog_memories, indent=2)}")
        assert len(dog_memories) > 0, "No dog memories found"
        assert dog_memories[0].get("readonly", False), "Memory should be marked as readonly in the connector"
        
        # Test deleting from any collection (should silently skip)
        deleted_count = await connector.delete_memories([cat_memory_id, dog_memory_id])
        print(f"Deleted count (should be 0 since all memories were skipped): {deleted_count}")
        assert deleted_count == 0, f"Expected to delete 0 memories, but got {deleted_count}"
        
        # Verify the memories still exist
        cat_memories_after = await connector.find_memories("cats", "cats")
        print(f"Cat memories after delete attempt: {json.dumps(cat_memories_after, indent=2)}")
        assert len(cat_memories_after) == len(cat_memories), "Cat memory should still exist"
        
        dog_memories_after = await connector.find_memories("dogs", "dogs")
        print(f"Dog memories after delete attempt: {json.dumps(dog_memories_after, indent=2)}")
        assert len(dog_memories_after) == len(dog_memories), "Dog memory should still exist"
        
        # Test finding memories across collections
        all_memories = await connector.find_memories_across_collections("pets")
        print(f"All memories: {json.dumps(all_memories, indent=2)}")
        
        # Verify readonly status in results for all collections
        for collection, memories in all_memories.items():
            for memory in memories:
                assert memory.get("readonly", False), f"{collection} memory should be marked as readonly in the connector"
        
        print("All protect all collections tests passed!")


async def test_delete_default_memories():
    """Test deleting memories from the default collection in multi-collection mode."""
    print("\nTesting deletion of memories from default collection in multi-collection mode...")
    
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
            collection_name="default",
            embedding_provider=provider,
            qdrant_local_path=temp_dir,
            multi_collection_mode=True,
            collection_prefix="test_",
        )
        
        # Store memories in the default collection
        default_memory_id1 = await connector.store_memory("This is the first default memory")
        print(f"Stored default memory 1 with ID: {default_memory_id1}")
        
        default_memory_id2 = await connector.store_memory("This is the second default memory")
        print(f"Stored default memory 2 with ID: {default_memory_id2}")
        
        # Store a memory in a different collection for comparison
        other_memory_id = await connector.store_memory("This is a memory in another collection", "other")
        print(f"Stored memory in 'other' collection with ID: {other_memory_id}")
        
        # Verify memories exist before deletion
        default_memories_before = await connector.find_memories("default memory")
        print(f"Default memories before deletion: {json.dumps(default_memories_before, indent=2)}")
        assert len(default_memories_before) == 2, f"Expected 2 default memories, found {len(default_memories_before)}"
        
        other_memories_before = await connector.find_memories("other", "other")
        print(f"Other memories before deletion: {json.dumps(other_memories_before, indent=2)}")
        assert len(other_memories_before) == 1, f"Expected 1 memory in 'other' collection, found {len(other_memories_before)}"
        
        # Delete one memory from the default collection
        deleted_count = await connector.delete_memories([default_memory_id1])
        print(f"Deleted {deleted_count} memory from default collection")
        assert deleted_count == 1, f"Expected to delete 1 memory, but deleted {deleted_count}"
        
        # Verify the memory was deleted
        default_memories_after = await connector.find_memories("default memory")
        print(f"Default memories after deletion: {json.dumps(default_memories_after, indent=2)}")
        assert len(default_memories_after) == 1, f"Expected 1 default memory after deletion, found {len(default_memories_after)}"
        
        # Verify the other collection is unaffected
        other_memories_after = await connector.find_memories("other", "other")
        print(f"Other memories after deletion: {json.dumps(other_memories_after, indent=2)}")
        assert len(other_memories_after) == 1, f"Expected 1 memory in 'other' collection, found {len(other_memories_after)}"
        
        # Delete the remaining default memory and the memory in the other collection
        deleted_count = await connector.delete_memories([default_memory_id2, other_memory_id])
        print(f"Deleted {deleted_count} memories")
        assert deleted_count == 2, f"Expected to delete 2 memories, but deleted {deleted_count}"
        
        # Verify all memories were deleted
        default_memories_final = await connector.find_memories("default memory")
        print(f"Default memories after final deletion: {json.dumps(default_memories_final, indent=2)}")
        assert len(default_memories_final) == 0, f"Expected 0 default memories, found {len(default_memories_final)}"
        
        other_memories_final = await connector.find_memories("other", "other")
        print(f"Other memories after final deletion: {json.dumps(other_memories_final, indent=2)}")
        assert len(other_memories_final) == 0, f"Expected 0 memories in 'other' collection, found {len(other_memories_final)}"
        
        print("All default memory deletion tests passed!")


async def test_readonly_collections():
    """Test read-only collections functionality."""
    print("\nTesting read-only collections functionality...")
    
    # Create a temporary directory for Qdrant local storage
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create embedding provider
        provider = create_embedding_provider(
            provider_type="fastembed",
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Create QdrantConnector with multi-collection mode enabled and read-only collections
        connector = QdrantConnector(
            qdrant_url=None,
            qdrant_api_key=None,
            collection_name="default",
            embedding_provider=provider,
            qdrant_local_path=temp_dir,
            multi_collection_mode=True,
            collection_prefix="test_",
            readonly_collections={"cats"}  # Make the "cats" collection read-only
        )
        
        # Test storing memories in different collections
        default_memory_id = await connector.store_memory("This is a default memory", None)
        print(f"Stored default memory with ID: {default_memory_id}")
        
        # Store a memory in the unprotected "dogs" collection
        dog_memory_id = await connector.store_memory("This is a memory about dogs", "dogs")
        print(f"Stored dog memory with ID: {dog_memory_id}")
        
        # Test storing in a read-only collection (should fail)
        try:
            cat_memory_id = await connector.store_memory("This is a memory about cats", "cats")
            assert False, "Should not be able to store memory in read-only collection"
        except ValueError as e:
            print(f"Expected error when storing in read-only collection: {e}")
            assert "read-only" in str(e), "Error message should mention read-only"
        
        # Create a memory in the cats collection by temporarily removing the read-only status
        connector._readonly_collections.remove("cats")
        cat_memory_id = await connector.store_memory("This is a memory about cats", "cats")
        print(f"Stored cat memory with ID: {cat_memory_id}")
        connector._readonly_collections.add("cats")
        
        # Test finding memories in read-only collection (should work)
        cat_memories = await connector.find_memories("cats", "cats")
        print(f"Cat memories: {json.dumps(cat_memories, indent=2)}")
        assert len(cat_memories) > 0, "No cat memories found"
        assert cat_memories[0].get("readonly", False), "Memory should be marked as readonly in the connector"
        
        # Test deleting memories in read-only collection (should fail)
        try:
            deleted_count = await connector.delete_memories([cat_memory_id])
            # This should not fail because delete_memories silently skips protected collections
            assert deleted_count == 0, "Should not delete any memories in read-only collection"
        except ValueError as e:
            print(f"Error when deleting from read-only collection: {e}")


async def test_readonly_all_collections():
    """Test making all collections read-only functionality."""
    print("\nTesting read-only all collections functionality...")
    
    # Create a temporary directory for Qdrant local storage
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create embedding provider
        provider = create_embedding_provider(
            provider_type="fastembed",
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Create QdrantConnector with multi-collection mode enabled and all collections read-only
        connector = QdrantConnector(
            qdrant_url=None,
            qdrant_api_key=None,
            collection_name="default",
            embedding_provider=provider,
            qdrant_local_path=temp_dir,
            multi_collection_mode=True,
            collection_prefix="test_",
            readonly_collections={"*"}  # Make all collections read-only
        )
        
        # Test storing memories in different collections (should all fail)
        try:
            default_memory_id = await connector.store_memory("This is a default memory", None)
            assert False, "Should not be able to store memory when all collections are read-only"
        except ValueError as e:
            print(f"Expected error when storing in read-only collection: {e}")
            assert "read-only" in str(e), "Error message should mention read-only"
        
        # Create a memory by temporarily removing the read-only status
        connector._readonly_collections.clear()
        default_memory_id = await connector.store_memory("This is a default memory", None)
        print(f"Stored default memory with ID: {default_memory_id}")
        cat_memory_id = await connector.store_memory("This is a memory about cats", "cats")
        print(f"Stored cat memory with ID: {cat_memory_id}")
        connector._readonly_collections.add("*")
        
        # Test finding memories (should work)
        default_memories = await connector.find_memories("default", None)
        print(f"Default memories: {json.dumps(default_memories, indent=2)}")
        assert len(default_memories) > 0, "No default memories found"
        assert default_memories[0].get("readonly", False), "Memory should be marked as readonly in the connector"
        
        cat_memories = await connector.find_memories("cats", "cats")
        print(f"Cat memories: {json.dumps(cat_memories, indent=2)}")
        assert len(cat_memories) > 0, "No cat memories found"
        assert cat_memories[0].get("readonly", False), "Memory should be marked as readonly in the connector"


if __name__ == "__main__":
    asyncio.run(test_multi_collection())
    asyncio.run(test_protected_collections())
    asyncio.run(test_protect_all_collections())
    asyncio.run(test_delete_default_memories())
    asyncio.run(test_readonly_collections())
    asyncio.run(test_readonly_all_collections()) 