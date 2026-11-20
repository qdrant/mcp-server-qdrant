"""End-to-end tests of the MCP tool layer, using FastMCP's in-memory client.

They cover how the configuration shapes the tool schemas (signature rewriting
in ``wrap_filters`` and ``make_partial_function``) and that the tools work
through a real MCP client.
"""

import uuid
import warnings
from importlib.metadata import version

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from mcp_server_qdrant.mcp_server import QdrantMCPServer
from mcp_server_qdrant.settings import (
    EmbeddingProviderSettings,
    FilterableField,
    QdrantSettings,
    ToolSettings,
)

FILTERABLE_FIELDS = [
    FilterableField(
        name="lang", description="Language", field_type="keyword", condition="=="
    ),
    FilterableField(
        name="year",
        description="Minimum year",
        field_type="integer",
        condition=">=",
        required=True,
    ),
    FilterableField(
        name="tags", description="Any of tags", field_type="keyword", condition="any"
    ),
]


@pytest.fixture(scope="module")
def embedding_provider_settings():
    return EmbeddingProviderSettings()


def make_server(embedding_provider_settings, **qdrant_settings) -> QdrantMCPServer:
    return QdrantMCPServer(
        tool_settings=ToolSettings(),
        qdrant_settings=QdrantSettings(QDRANT_URL=":memory:", **qdrant_settings),
        embedding_provider_settings=embedding_provider_settings,
    )


def schema_of(tools, name: str) -> dict:
    tool = next(t for t in tools if t.name == name)
    return tool.input_schema


def text_of(result) -> str:
    return " ".join(block.text for block in result.content)


async def test_default_config_exposes_collection_name(embedding_provider_settings):
    server = make_server(embedding_provider_settings)
    async with Client(server) as client:
        tools = await client.list_tools()
        assert {t.name for t in tools} == {
            "qdrant-find",
            "qdrant-store",
            "qdrant-list-collections",
        }

        find = schema_of(tools, "qdrant-find")
        assert set(find["properties"]) == {"query", "collection_name"}
        assert set(find["required"]) == {"query", "collection_name"}

        store = schema_of(tools, "qdrant-store")
        assert set(store["properties"]) == {
            "information",
            "collection_name",
            "metadata",
        }
        assert set(store["required"]) == {"information", "collection_name"}

        collection = f"test_{uuid.uuid4().hex}"
        await client.call_tool(
            "qdrant-store",
            {"information": "asyncio tips", "collection_name": collection},
        )
        result = await client.call_tool(
            "qdrant-find", {"query": "asyncio", "collection_name": collection}
        )
        assert "asyncio tips" in text_of(result)


async def test_fixed_collection_hides_collection_name(embedding_provider_settings):
    server = make_server(embedding_provider_settings, COLLECTION_NAME="fixed")
    async with Client(server) as client:
        tools = await client.list_tools()
        assert set(schema_of(tools, "qdrant-find")["properties"]) == {"query"}
        assert set(schema_of(tools, "qdrant-store")["properties"]) == {
            "information",
            "metadata",
        }

        result = await client.call_tool("qdrant-store", {"information": "hello"})
        assert "fixed" in text_of(result)


async def test_typed_filters_become_tool_parameters(embedding_provider_settings):
    server = make_server(
        embedding_provider_settings,
        COLLECTION_NAME="typed",
        filterable_fields=FILTERABLE_FIELDS,
    )
    async with Client(server) as client:
        tools = await client.list_tools()
        find = schema_of(tools, "qdrant-find")
        assert set(find["properties"]) == {"query", "lang", "year", "tags"}
        assert set(find["required"]) == {"query", "year"}

        await client.call_tool(
            "qdrant-store",
            {
                "information": "asyncio tips",
                "metadata": {"lang": "py", "year": 2024, "tags": ["a"]},
            },
        )

        hit = await client.call_tool(
            "qdrant-find", {"query": "asyncio", "year": 2020, "lang": "py"}
        )
        assert "asyncio tips" in text_of(hit)

        miss = await client.call_tool("qdrant-find", {"query": "asyncio", "year": 2030})
        assert "asyncio tips" not in text_of(miss)

        with pytest.raises(ToolError):
            await client.call_tool("qdrant-find", {"query": "asyncio"})
        with pytest.raises(ToolError):
            await client.call_tool("qdrant-find", {"query": "asyncio", "year": "abc"})


async def test_except_filter_condition(embedding_provider_settings):
    fields = [
        FilterableField(
            name="excluded_lang",
            description="Languages to exclude",
            field_type="keyword",
            condition="except",
        ),
    ]
    server = make_server(
        embedding_provider_settings,
        COLLECTION_NAME="except_test",
        filterable_fields=fields,
    )
    async with Client(server) as client:
        await client.call_tool(
            "qdrant-store",
            {
                "information": "python info",
                "metadata": {"excluded_lang": "python"},
            },
        )
        await client.call_tool(
            "qdrant-store",
            {
                "information": "rust info",
                "metadata": {"excluded_lang": "rust"},
            },
        )

        result = await client.call_tool(
            "qdrant-find",
            {"query": "info", "excluded_lang": ["rust"]},
        )
        assert "python info" in text_of(result)
        assert "rust info" not in text_of(result)


async def test_arbitrary_filter(embedding_provider_settings):
    server = make_server(
        embedding_provider_settings,
        COLLECTION_NAME="arbitrary",
        QDRANT_ALLOW_ARBITRARY_FILTER=True,
    )
    async with Client(server) as client:
        tools = await client.list_tools()
        assert set(schema_of(tools, "qdrant-find")["properties"]) == {
            "query",
            "query_filter",
        }

        await client.call_tool(
            "qdrant-store", {"information": "asyncio tips", "metadata": {"lang": "py"}}
        )
        query_filter = {"must": [{"key": "metadata.lang", "match": {"value": "py"}}]}
        result = await client.call_tool(
            "qdrant-find", {"query": "asyncio", "query_filter": query_filter}
        )
        assert "asyncio tips" in text_of(result)

        with pytest.raises(ToolError):
            await client.call_tool(
                "qdrant-find", {"query": "asyncio", "query_filter": {"must": "nope"}}
            )


async def test_read_only_hides_store(embedding_provider_settings):
    server = make_server(
        embedding_provider_settings, COLLECTION_NAME="ro", QDRANT_READ_ONLY=True
    )
    async with Client(server) as client:
        tools = await client.list_tools()
        assert {t.name for t in tools} == {"qdrant-find"}


async def test_server_reports_package_version(embedding_provider_settings):
    server = make_server(embedding_provider_settings, COLLECTION_NAME="v")
    expected = version("mcp-server-qdrant")

    # Clients on the 2026-07-28 protocol read it from server discovery.
    async with Client(server) as client:
        assert client.server_info.version == expected

    # Older clients read it from the initialize handshake.
    async with Client(server, mode="legacy") as client:
        assert client.initialize_result.server_info.version == expected


async def test_tools_do_not_use_deprecated_mcp_logging(embedding_provider_settings):
    server = make_server(embedding_provider_settings, COLLECTION_NAME="logging")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        async with Client(server) as client:
            await client.call_tool("qdrant-store", {"information": "hello"})
            await client.call_tool("qdrant-find", {"query": "hello"})

    deprecated = [
        w for w in caught if "logging capability is deprecated" in str(w.message)
    ]
    assert not deprecated


async def test_list_collections(embedding_provider_settings):
    server = make_server(embedding_provider_settings)
    names = [f"test_collection_{uuid.uuid4().hex}" for _ in range(2)]
    async with Client(server) as client:
        for name in names:
            await client.call_tool(
                "qdrant-store", {"information": "test", "collection_name": name}
            )
        listed = text_of(await client.call_tool("qdrant-list-collections", {}))
    for name in names:
        assert name in listed


async def test_list_collections_read_only(embedding_provider_settings):
    server = make_server(embedding_provider_settings, QDRANT_READ_ONLY=True)
    async with Client(server) as client:
        tools = {t.name for t in await client.list_tools()}
    assert tools == {"qdrant-find", "qdrant-list-collections"}


async def test_list_collections_hidden_with_default_collection(
    embedding_provider_settings,
):
    server = make_server(embedding_provider_settings, COLLECTION_NAME="default")
    async with Client(server) as client:
        tools = {t.name for t in await client.list_tools()}
    assert "qdrant-list-collections" not in tools
