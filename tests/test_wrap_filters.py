"""Tests for the signature rewriting performed by `wrap_filters`.

`wrap_filters` removes the raw `query_filter` argument of `qdrant-find` and
replaces it with one typed argument per configured metadata field, so that the
model can filter without knowing Qdrant's filter syntax. The generated
signature is what FastMCP turns into the tool JSON schema, so it is asserted
here directly.
"""

import inspect
from typing import Annotated, Any, Callable, Literal, Optional, get_args

import pytest
from pydantic import Field

from mcp_server_qdrant.common.filters import make_filter
from mcp_server_qdrant.common.wrap_filters import wrap_filters
from mcp_server_qdrant.settings import FilterableField

FieldType = Literal["keyword", "integer", "float", "boolean"]
Condition = Literal["==", "!=", ">", ">=", "<", "<=", "any", "except"]


def field(
    name: str,
    field_type: FieldType,
    condition: Condition | None = None,
    required: bool = False,
) -> FilterableField:
    """Build a ``FilterableField`` with only the given values customised."""
    return FilterableField(
        name=name,
        description=f"Description of {name}",
        field_type=field_type,
        condition=condition,
        required=required,
    )


def make_find(calls: list[dict[str, Any]]) -> Callable[..., list[str]]:
    """Build a stand-in for the inner `find` function of `qdrant-find`."""

    def find(
        query: Annotated[str, Field(description="What to search for")],
        collection_name: Annotated[str, Field(description="The collection")],
        query_filter: Optional[dict[str, Any]] = None,
    ) -> list[str]:
        calls.append(
            {
                "query": query,
                "collection_name": collection_name,
                "query_filter": query_filter,
            }
        )
        return ["found"]

    return find


def parameter(func: Callable, name: str) -> inspect.Parameter:
    return inspect.signature(func).parameters[name]


def described_type(func: Callable, name: str) -> Any:
    """Return the type described by an ``Annotated`` argument."""
    return get_args(parameter(func, name).annotation)[0]


def described_description(func: Callable, name: str) -> str | None:
    """Return the description attached to an ``Annotated`` argument."""
    return get_args(parameter(func, name).annotation)[1].description


class TestGeneratedSignature:
    def test_query_filter_is_replaced_by_the_filter_arguments(self) -> None:
        wrapped = wrap_filters(
            make_find([]), {"language": field("language", "keyword", "==")}
        )
        assert list(inspect.signature(wrapped).parameters) == [
            "query",
            "collection_name",
            "language",
        ]

    def test_original_arguments_are_kept(self) -> None:
        wrapped = wrap_filters(
            make_find([]), {"language": field("language", "keyword", "==")}
        )
        assert described_type(wrapped, "query") is str
        assert described_type(wrapped, "collection_name") is str

    def test_required_fields_come_before_optional_ones(self) -> None:
        # The dict order lists the optional field first, but an argument with a
        # default cannot precede one without a default.
        wrapped = wrap_filters(
            make_find([]),
            {
                "language": field("language", "keyword", "=="),
                "repo": field("repo", "keyword", "==", required=True),
            },
        )
        assert list(inspect.signature(wrapped).parameters) == [
            "query",
            "collection_name",
            "repo",
            "language",
        ]

    def test_required_field_has_no_default(self) -> None:
        wrapped = wrap_filters(
            make_find([]), {"repo": field("repo", "keyword", "==", required=True)}
        )
        assert parameter(wrapped, "repo").default is inspect.Parameter.empty

    def test_optional_field_defaults_to_none(self) -> None:
        wrapped = wrap_filters(
            make_find([]), {"language": field("language", "keyword", "==")}
        )
        assert parameter(wrapped, "language").default is None

    def test_field_description_is_exposed(self) -> None:
        wrapped = wrap_filters(
            make_find([]), {"language": field("language", "keyword", "==")}
        )
        assert described_description(wrapped, "language") == "Description of language"


class TestFilterArgumentTypes:
    @pytest.mark.parametrize(
        ("field_type", "expected_type"),
        [
            ("keyword", Optional[str]),
            ("integer", Optional[int]),
            ("float", Optional[float]),
            ("boolean", Optional[bool]),
        ],
    )
    def test_field_type_maps_to_python_type(
        self, field_type: FieldType, expected_type: Any
    ) -> None:
        wrapped = wrap_filters(
            make_find([]), {"memory": field("memory", field_type, "==")}
        )
        assert described_type(wrapped, "memory") == expected_type

    @pytest.mark.parametrize("condition", ["any", "except"])
    @pytest.mark.parametrize(
        ("field_type", "expected_type"),
        [("keyword", Optional[list[str]]), ("integer", Optional[list[int]])],
    )
    def test_list_conditions_use_list_types(
        self, condition: Condition, field_type: FieldType, expected_type: Any
    ) -> None:
        wrapped = wrap_filters(
            make_find([]), {"items": field("items", field_type, condition)}
        )
        assert described_type(wrapped, "items") == expected_type

    @pytest.mark.parametrize(
        ("field_type", "condition"),
        [
            ("float", "any"),
            ("float", "except"),
            ("boolean", "any"),
            ("boolean", "except"),
        ],
    )
    def test_list_conditions_reject_unsupported_types(
        self, field_type: FieldType, condition: Condition
    ) -> None:
        with pytest.raises(ValueError, match='Only "keyword" and "integer"'):
            wrap_filters(
                make_find([]), {"items": field("items", field_type, condition)}
            )

    def test_unsupported_field_type_raises(self) -> None:
        broken = FilterableField.model_construct(
            name="language",
            description="Description of language",
            field_type="uuid",
            condition="==",
            required=False,
        )
        with pytest.raises(ValueError, match="Unsupported field type: uuid"):
            wrap_filters(make_find([]), {"language": broken})

    def test_no_filterable_fields_just_drops_query_filter(self) -> None:
        wrapped = wrap_filters(make_find([]), {})
        assert list(inspect.signature(wrapped).parameters) == [
            "query",
            "collection_name",
        ]


class TestWrapperCall:
    def test_filter_arguments_build_the_query_filter(self) -> None:
        calls: list[dict[str, Any]] = []
        wrapped = wrap_filters(
            make_find(calls), {"language": field("language", "keyword", "==")}
        )

        assert wrapped(query="deps", collection_name="notes", language="python") == [
            "found"
        ]
        # The filter arguments must not be forwarded to the wrapped function.
        assert calls == [
            {
                "query": "deps",
                "collection_name": "notes",
                "query_filter": make_filter(
                    {"language": field("language", "keyword", "==")},
                    {"language": "python"},
                ),
            }
        ]

    def test_omitted_optional_arguments_produce_an_empty_filter(self) -> None:
        calls: list[dict[str, Any]] = []
        wrapped = wrap_filters(
            make_find(calls), {"language": field("language", "keyword", "==")}
        )
        wrapped(query="deps", collection_name="notes")

        assert calls[0]["query_filter"] == make_filter(
            {"language": field("language", "keyword", "==")}, {}
        )

    def test_required_field_is_forwarded(self) -> None:
        calls: list[dict[str, Any]] = []
        wrapped = wrap_filters(
            make_find(calls), {"repo": field("repo", "keyword", "==", required=True)}
        )
        wrapped(query="deps", collection_name="notes", repo="mcp-server-qdrant")

        assert calls[0]["query_filter"] == make_filter(
            {"repo": field("repo", "keyword", "==", required=True)},
            {"repo": "mcp-server-qdrant"},
        )


class TestWrapperIntrospection:
    def test_annotation_metadata_is_rebuilt(self) -> None:
        wrapped = wrap_filters(
            make_find([]), {"language": field("language", "keyword", "==")}
        )
        assert set(wrapped.__annotations__) == {
            "query",
            "collection_name",
            "language",
            "return",
        }
        assert wrapped.__annotations__["return"] == list[str]

    def test_original_function_is_preserved(self) -> None:
        wrapped = wrap_filters(
            make_find([]), {"language": field("language", "keyword", "==")}
        )
        assert wrapped.__name__ == "find"
        assert hasattr(wrapped, "__wrapped__")
