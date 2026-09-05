from types import SimpleNamespace

import pytest

from mcp_server_qdrant.embeddings.openai import OpenAIProvider


class FakeEmbeddings:
    """Records the payloads it receives and replays canned embeddings."""

    def __init__(self, vectors: list[list[float]]):
        self._vectors = vectors
        self.calls: list[dict] = []

    async def create(self, *, model: str, input: list[str]):
        self.calls.append({"model": model, "input": input})
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=index, embedding=vector)
                for index, vector in enumerate(self._vectors[: len(input)])
            ]
        )


def make_provider(vectors: list[list[float]], **kwargs) -> OpenAIProvider:
    provider = OpenAIProvider("test-model", base_url="http://localhost/v1", **kwargs)
    provider._client = SimpleNamespace(embeddings=FakeEmbeddings(vectors))
    return provider


@pytest.mark.asyncio
class TestOpenAIProvider:
    async def test_embed_documents(self):
        """Documents are embedded in a single request, preserving input order."""
        provider = make_provider([[1.0, 0.0], [0.0, 1.0]])

        embeddings = await provider.embed_documents(["first", "second"])

        assert embeddings == [[1.0, 0.0], [0.0, 1.0]]
        assert provider._client.embeddings.calls == [
            {"model": "test-model", "input": ["first", "second"]}
        ]

    async def test_embed_query(self):
        """A query is embedded on its own and returns a single vector."""
        provider = make_provider([[1.0, 0.0]])

        embedding = await provider.embed_query("a query")

        assert embedding == [1.0, 0.0]
        assert provider._client.embeddings.calls == [
            {"model": "test-model", "input": ["a query"]}
        ]

    async def test_out_of_order_response_is_sorted(self):
        """The API is only guaranteed to return an index, not a particular order."""
        provider = make_provider([[1.0], [2.0]])
        provider._client.embeddings.create = _reversed_create([[1.0], [2.0]])

        assert await provider.embed_documents(["a", "b"]) == [[1.0], [2.0]]

    async def test_prefixes_are_applied(self):
        """Asymmetric models need distinct query and document prefixes."""
        provider = make_provider(
            [[1.0]], query_prefix="Query: ", document_prefix="Passage: "
        )

        await provider.embed_documents(["doc"])
        await provider.embed_query("qry")

        assert [call["input"] for call in provider._client.embeddings.calls] == [
            ["Passage: doc"],
            ["Query: qry"],
        ]

    async def test_no_prefixes_by_default(self):
        """Without configuration the text is sent through untouched."""
        provider = make_provider([[1.0]])

        await provider.embed_query("qry")

        assert provider._client.embeddings.calls[0]["input"] == ["qry"]


class TestOpenAIProviderVectorMetadata:
    def test_vector_name_is_derived_from_the_model(self):
        provider = OpenAIProvider("Qwen/Qwen3-Embedding-8B")
        assert provider.get_vector_name() == "openai-qwen3-embedding-8b"

    def test_vector_name_can_be_overridden(self):
        provider = OpenAIProvider("test-model", vector_name="custom")
        assert provider.get_vector_name() == "custom"

    def test_configured_vector_size_is_used_without_a_request(self):
        """A configured size must not trigger a probe request to the API."""
        provider = OpenAIProvider("test-model", vector_size=1536)
        assert provider.get_vector_size() == 1536


def _reversed_create(vectors: list[list[float]]):
    async def create(*, model: str, input: list[str]):
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=index, embedding=vector)
                for index, vector in reversed(list(enumerate(vectors[: len(input)])))
            ]
        )

    return create
