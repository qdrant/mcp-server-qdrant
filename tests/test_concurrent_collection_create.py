import asyncio
import uuid
from unittest.mock import AsyncMock, patch

import pytest

from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider
from mcp_server_qdrant.qdrant import QdrantConnector


@pytest.fixture
async def embedding_provider():
    return FastEmbedProvider(model_name="sentence-transformers/all-MiniLM-L6-v2")


@pytest.mark.asyncio
async def test_ensure_collection_exists_survives_toctou_race(embedding_provider):
    """
    Regression test for the TOCTOU race in _ensure_collection_exists.

    Against a real Qdrant server (REST or gRPC), two concurrent store() calls
    into a not-yet-existing collection can both see collection_exists=False at
    the check step, and then both call create_collection. The second one is
    rejected by the server with an "already exists" error.

    Local ":memory:" mode does not reproduce this because its awaits do not
    yield inside the check-create window, so we force the interleaving with a
    mocked collection_exists that always returns False, mimicking the network
    race window a real server exposes.
    """
    collection_name = f"race_{uuid.uuid4().hex}"
    connector = QdrantConnector(
        qdrant_url=":memory:",
        qdrant_api_key=None,
        collection_name=collection_name,
        embedding_provider=embedding_provider,
    )

    with patch.object(
        connector._client, "collection_exists", new=AsyncMock(return_value=False)
    ):
        await asyncio.gather(
            connector._ensure_collection_exists(collection_name),
            connector._ensure_collection_exists(collection_name),
            connector._ensure_collection_exists(collection_name),
        )
