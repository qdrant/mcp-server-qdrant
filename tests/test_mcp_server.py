"""Tests for the tools exposed by `QdrantMCPServer`.

These tests exercise the real server, but with a stub embedding provider and a
Qdrant collection held in memory (`QDRANT_URL=":memory:"`), so they never reach
the network, download a model, or touch a user's database.
"""

import json
from typing import Any

import pytest
from fastmcp import FastMCP

from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.mcp_server import QdrantMCPServer
from mcp_server_qdrant.qdrant import Entry
from mcp_server_qdrant.settings import QdrantSettings, ToolSettings

VECTOR_SIZE = 4

# Local (in-memory) Qdrant cannot use payload indexes and says so on every
# collection creation. That is expected here; `test_qdrant_integration.py`
# covers the behaviour itself.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Payload indexes have no effect in the local Qdrant"
)

# Environment variables read by the settings models. They are cleared before
# each test so that the suite is not affected by the developer's shell.
SETTING_ENV_VARS = [
    "QDRANT_URL",
    "QDRANT_API_KEY",
    "QDRANT_LOCAL_PATH",
    "QDRANT_SEARCH_LIMIT",
    "QDRANT_READ_ONLY",
    "QDRANT_ALLOW_ARBITRARY_FILTER",
    "FILTERABLE_FIELDS",
    "COLLECTION_NAME",
    "TOOL_STORE_DESCRIPTION",
    "TOOL_FIND_DESCRIPTION",
]


class StubEmbeddingProvider(EmbeddingProvider):
    """Deterministic provider that avoids downloading an embedding model."""

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return [[0.1] * VECTOR_SIZE for _ in documents]

    async def embed_query(self, query: str) -> list[float]:
        return [0.1] * VECTOR_SIZE

    def get_vector_name(self) -> str:
        return "stub-vector"

    def get_vector_size(self) -> int:
        return VECTOR_SIZE


class FakeContext:
    """Minimal stand-in for the FastMCP context passed to every tool."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    async def debug(self, message: str) -> None:
        self.messages.append(message)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Clear every setting read from the environment, then use in-memory Qdrant."""
    for env_var in SETTING_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    return monkeypatch


def build_server() -> QdrantMCPServer:
    """Create a server using the settings currently in the environment."""
    return QdrantMCPServer(
        tool_settings=ToolSettings(),
        qdrant_settings=QdrantSettings(),
        embedding_provider=StubEmbeddingProvider(),
    )


async def get_tools(server: FastMCP) -> dict[str, Any]:
    return await server.get_tools()


async def test_only_the_two_expected_tools_are_registered(clean_env) -> None:
    tools = await get_tools(build_server())
    assert set(tools) == {"qdrant-find", "qdrant-store"}


async def test_read_only_mode_disables_the_store_tool(clean_env) -> None:
    clean_env.setenv("QDRANT_READ_ONLY", "true")
    tools = await get_tools(build_server())
    assert set(tools) == {"qdrant-find"}


class TestConstructor:
    def test_both_embedding_sources_are_rejected(self, clean_env) -> None:
        from mcp_server_qdrant.settings import EmbeddingProviderSettings

        with pytest.raises(ValueError, match="Cannot provide both"):
            QdrantMCPServer(
                tool_settings=ToolSettings(),
                qdrant_settings=QdrantSettings(),
                embedding_provider_settings=EmbeddingProviderSettings(),
                embedding_provider=StubEmbeddingProvider(),
            )

    def test_one_embedding_source_is_required(self, clean_env) -> None:
        with pytest.raises(ValueError, match="Must provide either"):
            QdrantMCPServer(
                tool_settings=ToolSettings(),
                qdrant_settings=QdrantSettings(),
            )


class TestCollectionNameArgument:
    async def test_absent_when_a_default_collection_is_configured(
        self, clean_env
    ) -> None:
        clean_env.setenv("COLLECTION_NAME", "memories")
        tools = await get_tools(build_server())
        for tool in tools.values():
            assert "collection_name" not in tool.parameters["properties"]

    async def test_required_when_no_default_collection_is_configured(
        self, clean_env
    ) -> None:
        tools = await get_tools(build_server())
        for tool in tools.values():
            assert "collection_name" in tool.parameters["properties"]
            assert "collection_name" in tool.parameters["required"]


class TestFindToolSchema:
    async def test_query_filter_is_hidden_by_default(self, clean_env) -> None:
        tools = await get_tools(build_server())
        assert "query_filter" not in tools["qdrant-find"].parameters["properties"]

    async def test_query_filter_is_exposed_when_arbitrary_filters_are_allowed(
        self, clean_env
    ) -> None:
        clean_env.setenv("QDRANT_ALLOW_ARBITRARY_FILTER", "true")
        tools = await get_tools(build_server())
        assert "query_filter" in tools["qdrant-find"].parameters["properties"]

    async def test_filterable_fields_become_tool_arguments(self, clean_env) -> None:
        clean_env.setenv(
            "FILTERABLE_FIELDS",
            json.dumps(
                [
                    {
                        "name": "language",
                        "description": "Programming language of the snippet",
                        "field_type": "keyword",
                        "condition": "==",
                    }
                ]
            ),
        )
        tools = await get_tools(build_server())
        properties = tools["qdrant-find"].parameters["properties"]

        assert "query_filter" not in properties
        assert properties["language"]["description"] == (
            "Programming language of the snippet"
        )

    async def test_optional_filterable_field_is_not_required(self, clean_env) -> None:
        clean_env.setenv(
            "FILTERABLE_FIELDS",
            json.dumps(
                [
                    {
                        "name": "language",
                        "description": "Programming language of the snippet",
                        "field_type": "keyword",
                        "condition": "==",
                    }
                ]
            ),
        )
        tools = await get_tools(build_server())
        assert "language" not in tools["qdrant-find"].parameters.get("required", [])

    async def test_required_filterable_field_is_required(self, clean_env) -> None:
        clean_env.setenv(
            "FILTERABLE_FIELDS",
            json.dumps(
                [
                    {
                        "name": "repo",
                        "description": "Repository the snippet came from",
                        "field_type": "keyword",
                        "condition": "==",
                        "required": True,
                    }
                ]
            ),
        )
        tools = await get_tools(build_server())
        assert "repo" in tools["qdrant-find"].parameters["required"]

    async def test_field_without_condition_is_not_exposed(self, clean_env) -> None:
        clean_env.setenv(
            "FILTERABLE_FIELDS",
            json.dumps(
                [
                    {
                        "name": "repo",
                        "description": "Repository the snippet came from",
                        "field_type": "keyword",
                    }
                ]
            ),
        )
        tools = await get_tools(build_server())
        properties = tools["qdrant-find"].parameters["properties"]

        assert "repo" not in properties
        assert "query_filter" not in properties

    async def test_filter_arguments_take_precedence_over_arbitrary_filters(
        self, clean_env
    ) -> None:
        # Declaring a field with a condition replaces `query_filter` with the
        # generated arguments, even when arbitrary filters are allowed.
        clean_env.setenv("QDRANT_ALLOW_ARBITRARY_FILTER", "true")
        clean_env.setenv(
            "FILTERABLE_FIELDS",
            json.dumps(
                [
                    {
                        "name": "language",
                        "description": "Programming language of the snippet",
                        "field_type": "keyword",
                        "condition": "==",
                    }
                ]
            ),
        )
        tools = await get_tools(build_server())
        properties = tools["qdrant-find"].parameters["properties"]

        assert "language" in properties
        assert "query_filter" not in properties


async def store_sample_entries(tools: dict[str, Any], ctx: FakeContext) -> None:
    """Store one entry per language used by the filtering tests."""
    store = tools["qdrant-store"].fn
    for information, language in (
        ("Use uv for deps", "python"),
        ("Use cargo for deps", "rust"),
    ):
        await store(ctx=ctx, information=information, metadata={"language": language})


class TestFindToolInvocation:
    async def test_equality_condition_selects_matching_entries(self, clean_env) -> None:
        clean_env.setenv("COLLECTION_NAME", "notes")
        clean_env.setenv(
            "FILTERABLE_FIELDS",
            json.dumps(
                [
                    {
                        "name": "language",
                        "description": "Programming language of the snippet",
                        "field_type": "keyword",
                        "condition": "==",
                    }
                ]
            ),
        )
        tools = await get_tools(build_server())
        ctx = FakeContext()
        await store_sample_entries(tools, ctx)

        results = await tools["qdrant-find"].fn(
            ctx=ctx, query="deps", language="python"
        )

        assert results == [
            "Results for the query 'deps'",
            "<entry><content>Use uv for deps</content>"
            '<metadata>{"language": "python"}</metadata></entry>',
        ]

    async def test_except_condition_excludes_matching_entries(self, clean_env) -> None:
        """`except` must round-trip through the tool, see `make_filter`."""
        clean_env.setenv("COLLECTION_NAME", "notes")
        clean_env.setenv(
            "FILTERABLE_FIELDS",
            json.dumps(
                [
                    {
                        "name": "language",
                        "description": "Programming language of the snippet",
                        "field_type": "keyword",
                        "condition": "except",
                    }
                ]
            ),
        )
        tools = await get_tools(build_server())
        ctx = FakeContext()
        await store_sample_entries(tools, ctx)

        results = await tools["qdrant-find"].fn(
            ctx=ctx, query="deps", language=["rust"]
        )

        assert results == [
            "Results for the query 'deps'",
            "<entry><content>Use uv for deps</content>"
            '<metadata>{"language": "python"}</metadata></entry>',
        ]

    async def test_arbitrary_filter_is_applied(self, clean_env) -> None:
        clean_env.setenv("COLLECTION_NAME", "notes")
        clean_env.setenv("QDRANT_ALLOW_ARBITRARY_FILTER", "true")
        tools = await get_tools(build_server())
        ctx = FakeContext()
        await store_sample_entries(tools, ctx)

        results = await tools["qdrant-find"].fn(
            ctx=ctx,
            query="deps",
            query_filter={
                "must": [{"key": "metadata.language", "match": {"value": "python"}}]
            },
        )

        assert results == [
            "Results for the query 'deps'",
            "<entry><content>Use uv for deps</content>"
            '<metadata>{"language": "python"}</metadata></entry>',
        ]


class TestFormatEntry:
    def test_entry_includes_content_and_metadata(self, clean_env) -> None:
        server = build_server()
        entry = Entry(
            content="Use uv for dependencies", metadata={"language": "python"}
        )
        assert server.format_entry(entry) == (
            "<entry><content>Use uv for dependencies</content>"
            '<metadata>{"language": "python"}</metadata></entry>'
        )

    def test_entry_without_metadata_has_an_empty_metadata_element(
        self, clean_env
    ) -> None:
        server = build_server()
        assert server.format_entry(Entry(content="No metadata here")) == (
            "<entry><content>No metadata here</content><metadata></metadata></entry>"
        )


class TestToolBehaviour:
    async def test_find_returns_nothing_for_an_empty_collection(
        self, clean_env
    ) -> None:
        clean_env.setenv("COLLECTION_NAME", "memories")
        tools = await get_tools(build_server())

        assert (
            await tools["qdrant-find"].fn(ctx=FakeContext(), query="anything") is None
        )

    async def test_stored_entries_can_be_found(self, clean_env) -> None:
        clean_env.setenv("COLLECTION_NAME", "memories")
        tools = await get_tools(build_server())
        ctx = FakeContext()

        await tools["qdrant-store"].fn(
            ctx=ctx,
            information="Use uv for dependencies",
            metadata={"language": "python"},
        )
        results = await tools["qdrant-find"].fn(ctx=ctx, query="dependencies")

        assert results is not None
        assert results[0] == "Results for the query 'dependencies'"
        assert results[1] == (
            "<entry><content>Use uv for dependencies</content>"
            '<metadata>{"language": "python"}</metadata></entry>'
        )

    async def test_store_reports_the_collection(self, clean_env) -> None:
        clean_env.setenv("COLLECTION_NAME", "memories")
        tools = await get_tools(build_server())

        message = await tools["qdrant-store"].fn(
            ctx=FakeContext(),
            information="Use uv for dependencies",
        )
        assert message == "Remembered: Use uv for dependencies in collection memories"

    @pytest.mark.filterwarnings("ignore:Payload indexes have no effect")
    async def test_find_applies_filter_arguments(self, clean_env) -> None:
        clean_env.setenv("COLLECTION_NAME", "memories")
        clean_env.setenv(
            "FILTERABLE_FIELDS",
            json.dumps(
                [
                    {
                        "name": "language",
                        "description": "Programming language of the snippet",
                        "field_type": "keyword",
                        "condition": "except",
                    }
                ]
            ),
        )
        tools = await get_tools(build_server())
        ctx = FakeContext()

        for information, language in [
            ("Use uv for dependencies", "python"),
            ("Use cargo for dependencies", "rust"),
        ]:
            await tools["qdrant-store"].fn(
                ctx=ctx, information=information, metadata={"language": language}
            )

        results = await tools["qdrant-find"].fn(
            ctx=ctx, query="dependencies", language=["rust"]
        )

        assert results is not None
        assert len(results) == 2
        assert "Use cargo for dependencies" not in "".join(results)
