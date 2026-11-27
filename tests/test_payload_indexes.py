"""
Payload indexes for filterable fields must exist even when the collection already
did before the server first used it. Local mode ignores payload indexes, so these
tests use it with ``_is_local`` turned off and record ``create_payload_index``
calls instead of checking the collection.
"""

import logging
import uuid
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from qdrant_client import models
from qdrant_client.http.exceptions import UnexpectedResponse

from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider
from mcp_server_qdrant.qdrant import Entry, QdrantConnector

INDEXES = {
    "metadata.lang": models.PayloadSchemaType.KEYWORD,
    "metadata.year": models.PayloadSchemaType.INTEGER,
}


@pytest.fixture(scope="module")
def embedding_provider():
    return FastEmbedProvider(model_name="sentence-transformers/all-MiniLM-L6-v2")


async def make_connector(embedding_provider, *, remote: bool = True):
    """A connector whose collection already exists, created without any index."""
    collection_name = f"indexes_{uuid.uuid4().hex}"
    connector = QdrantConnector(
        qdrant_url=":memory:",
        qdrant_api_key=None,
        collection_name=collection_name,
        embedding_provider=embedding_provider,
        field_indexes=INDEXES,
    )
    connector._is_local = not remote
    await connector._client.create_collection(
        collection_name,
        vectors_config={
            embedding_provider.get_vector_name(): models.VectorParams(
                size=embedding_provider.get_vector_size(),
                distance=models.Distance.COSINE,
            )
        },
    )
    return connector


def forbidden() -> UnexpectedResponse:
    return UnexpectedResponse(
        status_code=403,
        reason_phrase="Forbidden",
        content=b'{"status":{"error":"Forbidden"}}',
        headers=httpx.Headers(),
    )


async def test_store_creates_missing_indexes(embedding_provider):
    connector = await make_connector(embedding_provider)
    with patch.object(
        connector._client, "create_payload_index", new=AsyncMock()
    ) as create_index:
        await connector.store(Entry(content="hello"))

    created = {call.kwargs["field_name"] for call in create_index.await_args_list}
    assert created == set(INDEXES)


async def test_search_creates_missing_indexes(embedding_provider):
    """In read-only mode, search is the only call that reaches the collection."""
    connector = await make_connector(embedding_provider)
    with patch.object(
        connector._client, "create_payload_index", new=AsyncMock()
    ) as create_index:
        await connector.search("hello")

    created = {call.kwargs["field_name"] for call in create_index.await_args_list}
    assert created == set(INDEXES)


async def test_existing_indexes_are_not_created_again(embedding_provider):
    connector = await make_connector(embedding_provider)
    info = await connector._client.get_collection(connector._default_collection_name)
    info.payload_schema = {
        "metadata.lang": models.PayloadIndexInfo(
            data_type=models.PayloadSchemaType.KEYWORD, points=0
        )
    }
    with (
        patch.object(connector, "_get_collection", new=AsyncMock(return_value=info)),
        patch.object(
            connector._client, "create_payload_index", new=AsyncMock()
        ) as create_index,
    ):
        await connector.search("hello")

    created = [call.kwargs["field_name"] for call in create_index.await_args_list]
    assert created == ["metadata.year"]


async def test_local_mode_does_not_create_indexes(embedding_provider):
    """Local mode never reports indexes, so creating them would warn on every call."""
    connector = await make_connector(embedding_provider, remote=False)
    with patch.object(
        connector._client, "create_payload_index", new=AsyncMock()
    ) as create_index:
        await connector.store(Entry(content="hello"))
        await connector.search("hello")

    create_index.assert_not_awaited()


async def test_key_without_write_access_logs_a_warning(embedding_provider, caplog):
    connector = await make_connector(embedding_provider)
    await connector._client.upsert(
        connector._default_collection_name,
        points=[
            models.PointStruct(
                id=uuid.uuid4().hex,
                vector={
                    embedding_provider.get_vector_name(): (
                        await embedding_provider.embed_documents(["hello"])
                    )[0]
                },
                payload={"document": "hello"},
            )
        ],
    )
    with (
        patch.object(
            connector._client,
            "create_payload_index",
            new=AsyncMock(side_effect=forbidden()),
        ) as create_index,
        caplog.at_level(logging.WARNING),
    ):
        results = await connector.search("hello")

    assert [entry.content for entry in results] == ["hello"]
    # The other index would be refused the same way, so it isn't attempted.
    assert create_index.await_count == 1
    assert "not allowed to create one" in caplog.text
    assert "metadata.lang" in caplog.text


async def test_other_index_errors_are_raised(embedding_provider):
    connector = await make_connector(embedding_provider)
    error = UnexpectedResponse(
        status_code=500,
        reason_phrase="Internal Server Error",
        content=b"{}",
        headers=httpx.Headers(),
    )
    with (
        patch.object(
            connector._client, "create_payload_index", new=AsyncMock(side_effect=error)
        ),
        pytest.raises(UnexpectedResponse),
    ):
        await connector.search("hello")
