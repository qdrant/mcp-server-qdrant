from typing import Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings

from mcp_server_qdrant.embeddings.types import EmbeddingProviderType

DEFAULT_TOOL_STORE_DESCRIPTION = (
    "Keep the memory for later use, when you are asked to remember something."
)
DEFAULT_TOOL_FIND_DESCRIPTION = (
    "Look up memories in Qdrant. Use this tool when you need to: \n"
    " - Find memories by their content \n"
    " - Access memories for further analysis \n"
    " - Get some personal information about the user"
)

METADATA_PATH = "metadata"


class ToolSettings(BaseSettings):
    """
    Configuration for all the tools.
    """

    tool_store_description: str = Field(
        default=DEFAULT_TOOL_STORE_DESCRIPTION,
        validation_alias="TOOL_STORE_DESCRIPTION",
    )
    tool_find_description: str = Field(
        default=DEFAULT_TOOL_FIND_DESCRIPTION,
        validation_alias="TOOL_FIND_DESCRIPTION",
    )


class EmbeddingProviderSettings(BaseSettings):
    """
    Configuration for the embedding provider.
    """

    provider_type: EmbeddingProviderType = Field(
        default=EmbeddingProviderType.FASTEMBED,
        validation_alias="EMBEDDING_PROVIDER",
    )
    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        validation_alias="EMBEDDING_MODEL",
    )


# The conditions that `make_filter` can turn into a Qdrant filter, for each field type.
SUPPORTED_CONDITIONS: dict[str, tuple[str, ...]] = {
    "keyword": ("==", "!=", "any", "except"),
    "integer": ("==", "!=", ">", ">=", "<", "<=", "any", "except"),
    "float": (">", ">=", "<", "<="),
    "boolean": ("==", "!="),
}


class FilterableField(BaseModel):
    name: str = Field(description="The name of the field payload field to filter on")
    description: str = Field(
        description="A description for the field used in the tool description"
    )
    field_type: Literal["keyword", "integer", "float", "boolean"] = Field(
        description="The type of the field"
    )
    condition: Literal["==", "!=", ">", ">=", "<", "<=", "any", "except"] | None = (
        Field(
            default=None,
            description=(
                "The condition to use for the filter. If not provided, the field will be indexed, but no "
                "filter argument will be exposed to MCP tool."
            ),
        )
    )
    required: bool = Field(
        default=False,
        description="Whether the field is required for the filter.",
    )

    @model_validator(mode="after")
    def check_condition_supported(self) -> "FilterableField":
        """
        Reject a condition the field type does not support when the settings
        load, instead of exposing a filter that fails every time it is used.
        """
        supported = SUPPORTED_CONDITIONS[self.field_type]
        if self.condition is not None and self.condition not in supported:
            raise ValueError(
                f"Condition '{self.condition}' is not supported for {self.field_type} "
                f"field '{self.name}'. Supported conditions: {', '.join(supported)}."
            )
        return self


class QdrantSettings(BaseSettings):
    """
    Configuration for the Qdrant connector.
    """

    location: str | None = Field(default=None, validation_alias="QDRANT_URL")
    api_key: str | None = Field(default=None, validation_alias="QDRANT_API_KEY")
    collection_name: str | None = Field(
        default=None, validation_alias="COLLECTION_NAME"
    )
    local_path: str | None = Field(default=None, validation_alias="QDRANT_LOCAL_PATH")
    search_limit: int = Field(default=10, validation_alias="QDRANT_SEARCH_LIMIT")
    read_only: bool = Field(default=False, validation_alias="QDRANT_READ_ONLY")

    filterable_fields: list[FilterableField] | None = Field(default=None)

    allow_arbitrary_filter: bool = Field(
        default=False, validation_alias="QDRANT_ALLOW_ARBITRARY_FILTER"
    )

    def filterable_fields_dict(self) -> dict[str, FilterableField]:
        if self.filterable_fields is None:
            return {}
        return {field.name: field for field in self.filterable_fields}

    def filterable_fields_dict_with_conditions(self) -> dict[str, FilterableField]:
        if self.filterable_fields is None:
            return {}
        return {
            field.name: field
            for field in self.filterable_fields
            if field.condition is not None
        }

    @model_validator(mode="after")
    def check_local_path_conflict(self) -> "QdrantSettings":
        if self.local_path:
            if self.location is not None or self.api_key is not None:
                raise ValueError(
                    "If 'local_path' is set, 'location' and 'api_key' must be None."
                )
        return self
