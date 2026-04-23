import inspect
from functools import wraps
from typing import Annotated, Callable, Optional

from pydantic import Field

from mcp_server_qdrant.common.filters import make_filter
from mcp_server_qdrant.settings import FilterableField


def wrap_filters(
    original_func: Callable, filterable_fields: dict[str, FilterableField]
) -> Callable:
    """
    Wraps the original_func function: replaces `filter` parameter with multiple parameters defined by `filterable_fields`.
    """

    sig = inspect.signature(original_func)

    @wraps(original_func)
    def wrapper(*args, **kwargs):
        # Start with fixed values
        filter_values = {}

        for field_name in filterable_fields:
            if field_name in kwargs:
                filter_values[field_name] = kwargs.pop(field_name)

        query_filter = make_filter(filterable_fields, filter_values)

        return original_func(**kwargs, query_filter=query_filter)

    # Replace `query_filter` signature with parameters from `filterable_fields`

    param_names = []

    for param_name in sig.parameters:
        if param_name == "query_filter":
            continue
        param_names.append(param_name)

    new_params = [sig.parameters[param_name] for param_name in param_names]
    required_new_params = []
    optional_new_params = []

    # Create a new signature parameters from `filterable_fields`
    for field in filterable_fields.values():
        field_name = field.name
        field_type: type
        if field.field_type == "keyword":
            field_type = str
        elif field.field_type == "integer":
            field_type = int
        elif field.field_type == "float":
            field_type = float
        elif field.field_type == "boolean":
            field_type = bool
        else:
            raise ValueError(f"Unsupported field type: {field.field_type}")

        if field.condition in {"any", "except"}:
            if field_type not in {str, int}:
                raise ValueError(
                    f'Only "keyword" and "integer" types are supported for "{field.condition}" condition'
                )
            field_type = list[field_type]  # type: ignore

        if field.required:
            annotation = Annotated[field_type, Field(description=field.description)]  # type: ignore
            parameter = inspect.Parameter(
                name=field_name,
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=annotation,
            )
            required_new_params.append(parameter)
        else:
            annotation = Annotated[  # type: ignore
                Optional[field_type], Field(description=field.description)
            ]
            parameter = inspect.Parameter(
                name=field_name,
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=None,
                annotation=annotation,
            )
            optional_new_params.append(parameter)

    new_params.extend(required_new_params)
    new_params.extend(optional_new_params)

    # Set the new __signature__ for introspection
    new_signature = sig.replace(parameters=new_params)
    wrapper.__signature__ = new_signature  # type: ignore

    # Set the new __annotations__ for introspection
    new_annotations = {}
    for param in new_signature.parameters.values():
        if param.annotation != inspect.Parameter.empty:
            new_annotations[param.name] = param.annotation

    # Add return type annotation if it exists
    if new_signature.return_annotation != inspect.Parameter.empty:
        new_annotations["return"] = new_signature.return_annotation

    wrapper.__annotations__ = new_annotations

    return wrapper


if __name__ == "__main__":
    from pydantic._internal._typing_extra import get_function_type_hints
    from qdrant_client import models

    def find(
        query: Annotated[str, Field(description="What to search for")],
        collection_name: Annotated[
            str, Field(description="The collection to search in")
        ],
        query_filter: Optional[models.Filter] = None,
    ) -> list[str]:
        print("query", query)
        print("collection_name", collection_name)
        print("query_filter", query_filter)
        return ["mypy rules"]

    wrapped_find = wrap_filters(
        find,
        {
            "color": FilterableField(
                name="color",
                description="The color of the object",
                field_type="keyword",
                condition="==",
            ),
            "size": FilterableField(
                name="size",
                description="The size of the object",
                field_type="keyword",
                condition="==",
                required=True,
            ),
        },
    )

    wrapped_find(query="dress", collection_name="test", color="red")

    print("get_function_type_hints(find)", get_function_type_hints(find))
    print(
        "get_function_type_hints(wrapped_find)", get_function_type_hints(wrapped_find)
    )
