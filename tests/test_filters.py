"""Unit tests for the filter helpers used by the `qdrant-find` tool.

The tool exposes one MCP argument per configured metadata field, and
`make_filter`/`make_indexes` turn those arguments into the Qdrant payload
indexes and the `query_filter` accepted by the Qdrant client.
"""

import json
from typing import Any, Literal, cast

import pytest
from qdrant_client import models

from mcp_server_qdrant.common.filters import make_filter, make_indexes
from mcp_server_qdrant.settings import METADATA_PATH, FilterableField

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


def build_filter(
    filterable_fields: dict[str, FilterableField], values: dict[str, Any]
) -> models.Filter:
    """Run ``make_filter`` and rebuild the model, like `qdrant-find` does."""
    return models.Filter(**make_filter(filterable_fields, values))


class TestMakeIndexes:
    @pytest.mark.parametrize(
        ("field_type", "schema_type"),
        [
            ("keyword", models.PayloadSchemaType.KEYWORD),
            ("integer", models.PayloadSchemaType.INTEGER),
            ("float", models.PayloadSchemaType.FLOAT),
            ("boolean", models.PayloadSchemaType.BOOL),
        ],
    )
    def test_field_type_maps_to_payload_schema(
        self, field_type: FieldType, schema_type: models.PayloadSchemaType
    ) -> None:
        indexes = make_indexes({"memory": field("memory", field_type)})
        assert indexes == {f"{METADATA_PATH}.memory": schema_type}

    def test_index_names_are_prefixed_with_metadata_path(self) -> None:
        indexes = make_indexes({"language": field("language", "keyword")})
        assert list(indexes) == [f"{METADATA_PATH}.language"]

    def test_no_fields_means_no_indexes(self) -> None:
        assert make_indexes({}) == {}

    def test_field_without_condition_is_still_indexed(self) -> None:
        # Fields without a condition are not exposed as tool arguments, but
        # they are still indexed so that arbitrary filters stay fast.
        indexes = make_indexes({"repo": field("repo", "keyword")})
        assert indexes == {f"{METADATA_PATH}.repo": models.PayloadSchemaType.KEYWORD}

    def test_every_field_is_indexed(self) -> None:
        indexes = make_indexes(
            {
                "language": field("language", "keyword"),
                "stars": field("stars", "integer", ">="),
                "score": field("score", "float", "<"),
                "archived": field("archived", "boolean", "=="),
            }
        )
        assert indexes == {
            f"{METADATA_PATH}.language": models.PayloadSchemaType.KEYWORD,
            f"{METADATA_PATH}.stars": models.PayloadSchemaType.INTEGER,
            f"{METADATA_PATH}.score": models.PayloadSchemaType.FLOAT,
            f"{METADATA_PATH}.archived": models.PayloadSchemaType.BOOL,
        }

    def test_unsupported_field_type_raises(self) -> None:
        # `FilterableField` validates the field type, so bypass the validation
        # to cover the defensive branch of `make_indexes`.
        broken = FilterableField.model_construct(
            name="language",
            description="Description of language",
            field_type="uuid",
            condition=None,
            required=False,
        )
        with pytest.raises(ValueError, match="Unsupported field type uuid"):
            make_indexes({"language": broken})


class TestKeywordFilters:
    def test_equality_generates_a_must_condition(self) -> None:
        query_filter = build_filter(
            {"language": field("language", "keyword", "==")}, {"language": "python"}
        )
        assert query_filter.must == [
            models.FieldCondition(
                key=f"{METADATA_PATH}.language",
                match=models.MatchValue(value="python"),
            )
        ]
        assert query_filter.must_not == []

    def test_inequality_generates_a_must_not_condition(self) -> None:
        query_filter = build_filter(
            {"language": field("language", "keyword", "!=")}, {"language": "rust"}
        )
        assert query_filter.must == []
        assert query_filter.must_not == [
            models.FieldCondition(
                key=f"{METADATA_PATH}.language",
                match=models.MatchValue(value="rust"),
            )
        ]

    def test_any_generates_a_match_any_condition(self) -> None:
        query_filter = build_filter(
            {"language": field("language", "keyword", "any")},
            {"language": ["python", "rust"]},
        )
        assert query_filter.must == [
            models.FieldCondition(
                key=f"{METADATA_PATH}.language",
                match=models.MatchAny(any=["python", "rust"]),
            )
        ]

    def test_except_generates_a_match_except_condition(self) -> None:
        query_filter = build_filter(
            {"language": field("language", "keyword", "except")},
            {"language": ["rust"]},
        )
        assert query_filter.must == [
            models.FieldCondition(
                key=f"{METADATA_PATH}.language",
                match=models.MatchExcept(**{"except": ["rust"]}),
            )
        ]

    def test_unsupported_condition_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid condition > for keyword field"):
            build_filter(
                {"language": field("language", "keyword", ">")},
                {"language": "python"},
            )


class TestIntegerFilters:
    def test_equality_generates_a_must_condition(self) -> None:
        query_filter = build_filter(
            {"stars": field("stars", "integer", "==")}, {"stars": 5}
        )
        assert query_filter.must == [
            models.FieldCondition(
                key=f"{METADATA_PATH}.stars", match=models.MatchValue(value=5)
            )
        ]

    def test_inequality_generates_a_must_not_condition(self) -> None:
        query_filter = build_filter(
            {"stars": field("stars", "integer", "!=")}, {"stars": 5}
        )
        assert query_filter.must_not == [
            models.FieldCondition(
                key=f"{METADATA_PATH}.stars", match=models.MatchValue(value=5)
            )
        ]

    @pytest.mark.parametrize(
        ("condition", "expected_range"),
        [
            (">", models.Range(gt=5)),
            (">=", models.Range(gte=5)),
            ("<", models.Range(lt=5)),
            ("<=", models.Range(lte=5)),
        ],
    )
    def test_range_conditions(
        self, condition: Condition, expected_range: models.Range
    ) -> None:
        query_filter = build_filter(
            {"stars": field("stars", "integer", condition)}, {"stars": 5}
        )
        assert query_filter.must == [
            models.FieldCondition(key=f"{METADATA_PATH}.stars", range=expected_range)
        ]

    def test_any_generates_a_match_any_condition(self) -> None:
        query_filter = build_filter(
            {"stars": field("stars", "integer", "any")}, {"stars": [1, 2]}
        )
        assert query_filter.must == [
            models.FieldCondition(
                key=f"{METADATA_PATH}.stars", match=models.MatchAny(any=[1, 2])
            )
        ]

    def test_except_generates_a_match_except_condition(self) -> None:
        query_filter = build_filter(
            {"stars": field("stars", "integer", "except")}, {"stars": [3]}
        )
        assert query_filter.must == [
            models.FieldCondition(
                key=f"{METADATA_PATH}.stars",
                match=models.MatchExcept(**{"except": [3]}),
            )
        ]


class TestFloatFilters:
    @pytest.mark.parametrize(
        ("condition", "expected_range"),
        [
            (">", models.Range(gt=1.5)),
            (">=", models.Range(gte=1.5)),
            ("<", models.Range(lt=1.5)),
            ("<=", models.Range(lte=1.5)),
        ],
    )
    def test_range_conditions(
        self, condition: Condition, expected_range: models.Range
    ) -> None:
        query_filter = build_filter(
            {"score": field("score", "float", condition)}, {"score": 1.5}
        )
        assert query_filter.must == [
            models.FieldCondition(key=f"{METADATA_PATH}.score", range=expected_range)
        ]

    @pytest.mark.parametrize("condition", ["==", "!=", "any", "except"])
    def test_non_range_conditions_are_rejected(self, condition: Condition) -> None:
        with pytest.raises(ValueError, match="Only range comparisons"):
            build_filter({"score": field("score", "float", condition)}, {"score": 1.5})


class TestBooleanFilters:
    def test_equality_generates_a_must_condition(self) -> None:
        query_filter = build_filter(
            {"archived": field("archived", "boolean", "==")}, {"archived": True}
        )
        assert query_filter.must == [
            models.FieldCondition(
                key=f"{METADATA_PATH}.archived", match=models.MatchValue(value=True)
            )
        ]

    def test_inequality_generates_a_must_not_condition(self) -> None:
        query_filter = build_filter(
            {"archived": field("archived", "boolean", "!=")}, {"archived": False}
        )
        assert query_filter.must_not == [
            models.FieldCondition(
                key=f"{METADATA_PATH}.archived", match=models.MatchValue(value=False)
            )
        ]

    def test_unsupported_condition_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid condition > for boolean field"):
            build_filter(
                {"archived": field("archived", "boolean", ">")}, {"archived": True}
            )


class TestFilterValues:
    def test_no_values_generates_an_empty_filter(self) -> None:
        query_filter = build_filter(
            {"language": field("language", "keyword", "==")}, {}
        )
        assert query_filter.must == []
        assert query_filter.must_not == []

    def test_optional_field_with_no_value_is_skipped(self) -> None:
        query_filter = build_filter(
            {"language": field("language", "keyword", "==")}, {"language": None}
        )
        assert query_filter.must == []

    def test_required_field_with_no_value_raises(self) -> None:
        with pytest.raises(ValueError, match="Field language is required"):
            build_filter(
                {"language": field("language", "keyword", "==", required=True)},
                {"language": None},
            )

    def test_value_for_unknown_field_raises(self) -> None:
        with pytest.raises(ValueError, match="Field language is not a filterable"):
            build_filter(
                {"repo": field("repo", "keyword", "==")}, {"language": "python"}
            )

    def test_field_without_condition_contributes_no_condition(self) -> None:
        # Only fields with a condition take part in the filter; the others are
        # declared for indexing purposes only.
        query_filter = build_filter(
            {"repo": field("repo", "keyword")}, {"repo": "mcp-server-qdrant"}
        )
        assert query_filter.must == []
        assert query_filter.must_not == []

    def test_multiple_fields_generate_multiple_conditions(self) -> None:
        query_filter = build_filter(
            {
                "language": field("language", "keyword", "=="),
                "archived": field("archived", "boolean", "!="),
            },
            {"language": "python", "archived": True},
        )
        assert query_filter.must == [
            models.FieldCondition(
                key=f"{METADATA_PATH}.language",
                match=models.MatchValue(value="python"),
            )
        ]
        assert query_filter.must_not == [
            models.FieldCondition(
                key=f"{METADATA_PATH}.archived", match=models.MatchValue(value=True)
            )
        ]

    def test_unsupported_field_type_raises(self) -> None:
        broken = FilterableField.model_construct(
            name="language",
            description="Description of language",
            field_type="uuid",
            condition="==",
            required=False,
        )
        with pytest.raises(ValueError, match="Unsupported field type uuid"):
            build_filter({"language": broken}, {"language": "python"})


class TestFilterSerialisation:
    """`make_filter` returns a dict that `qdrant-find` deserialises again.

    Regression tests for the fact that `MatchExcept` must be serialised with
    the Qdrant alias (`except`) rather than its Python field name (`except_`),
    otherwise every `qdrant-find` call using an `except` condition fails.
    """

    def test_match_except_uses_the_qdrant_alias(self) -> None:
        query_filter = make_filter(
            {"language": field("language", "keyword", "except")},
            {"language": ["rust"]},
        )
        serialised = json.dumps(query_filter)
        assert '"except"' in serialised
        assert "except_" not in serialised

    def test_every_condition_type_can_be_deserialised_again(self) -> None:
        filterable_fields = {
            "keyword_eq": field("keyword_eq", "keyword", "=="),
            "keyword_neq": field("keyword_neq", "keyword", "!="),
            "keyword_any": field("keyword_any", "keyword", "any"),
            "keyword_except": field("keyword_except", "keyword", "except"),
            "integer_except": field("integer_except", "integer", "except"),
            "integer_gte": field("integer_gte", "integer", ">="),
            "float_lt": field("float_lt", "float", "<"),
            "boolean_eq": field("boolean_eq", "boolean", "=="),
        }
        values: dict[str, Any] = {
            "keyword_eq": "a",
            "keyword_neq": "b",
            "keyword_any": ["a", "b"],
            "keyword_except": ["c"],
            "integer_except": [1],
            "integer_gte": 2,
            "float_lt": 3.5,
            "boolean_eq": True,
        }

        query_filter = build_filter(filterable_fields, values)

        # `Filter.must` accepts several condition types; `make_filter` only ever
        # produces `FieldCondition`, and a wrong type would fail the assertions
        # below rather than pass silently.
        must = cast(list[models.FieldCondition], query_filter.must or [])
        must_not = cast(list[models.FieldCondition], query_filter.must_not or [])

        assert {condition.key for condition in must} == {
            f"{METADATA_PATH}.keyword_eq",
            f"{METADATA_PATH}.keyword_any",
            f"{METADATA_PATH}.keyword_except",
            f"{METADATA_PATH}.integer_except",
            f"{METADATA_PATH}.integer_gte",
            f"{METADATA_PATH}.float_lt",
            f"{METADATA_PATH}.boolean_eq",
        }
        assert [condition.key for condition in must_not] == [
            f"{METADATA_PATH}.keyword_neq"
        ]
