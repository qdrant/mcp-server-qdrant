from typing import List, Literal

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


class CustomModelSettings(BaseModel):
    """
    Configuration for custom FastEmbed models using add_custom_model.
    """
    
    model_name: str | None = Field(
        default=None, 
        description="The name/identifier for the custom model"
    )
    hf_model_id: str | None = Field(
        default=None, 
        description="HuggingFace model ID for the custom model"
    )
    model_url: str | None = Field(
        default=None,
        description="Direct URL to model files (alternative to HuggingFace)"
    )
    pooling_type: str = Field(
        default="MEAN",
        description="Pooling type: MEAN, CLS, MAX"
    )
    normalization: bool = Field(
        default=True,
        description="Whether to normalize embeddings"
    )
    vector_dimension: int | None = Field(
        default=None,
        description="Dimension of the embedding vectors"
    )
    model_file: str | None = Field(
        default=None,
        description="Path to the model file within the model directory"
    )
    additional_files: List[str] | None = Field(
        default=None,
        description="Path to additional files within the model directory"
    )


    @model_validator(mode="after")
    def check_model_source(self) -> "CustomModelSettings":
        if not self.hf_model_id and not self.model_url:
            raise ValueError(
                "Either 'hf_model_id' or 'model_url' must be provided for custom model"
            )
        if self.hf_model_id and self.model_url:
            raise ValueError(
                "Only one of 'hf_model_id' or 'model_url' should be provided"
            )
        return self


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
    
    # Custom model configuration
    use_custom_model: bool = Field(
        default=False,
        validation_alias="EMBEDDING_USE_CUSTOM_MODEL",
    )
    custom_model_name: str | None = Field(
        default=None,
        validation_alias="EMBEDDING_CUSTOM_MODEL_NAME",
    )
    custom_hf_model_id: str | None = Field(
        default=None,
        validation_alias="EMBEDDING_CUSTOM_HF_MODEL_ID",
    )
    custom_model_url: str | None = Field(
        default=None,
        validation_alias="EMBEDDING_CUSTOM_MODEL_URL",
    )
    custom_pooling_type: str = Field(
        default="MEAN",
        validation_alias="EMBEDDING_CUSTOM_POOLING_TYPE",
    )
    custom_normalization: bool = Field(
        default=True,
        validation_alias="EMBEDDING_CUSTOM_NORMALIZATION",
    )
    custom_vector_dimension: int | None = Field(
        default=None,
        validation_alias="EMBEDDING_CUSTOM_VECTOR_DIMENSION",
    )
    custom_model_file: str = Field(
        default="onnx/model.onnx",
        validation_alias="EMBEDDING_CUSTOM_MODEL_FILE",
    )
    custom_additional_files: str | None = Field(
        default=None,
        validation_alias="EMBEDDING_CUSTOM_ADDITIONAL_FILES",
    )

    def get_custom_model_settings(self) -> CustomModelSettings | None:
        """Get custom model settings if custom model is enabled."""
        if not self.use_custom_model:
            return None
        
        if not self.custom_model_name:
            raise ValueError("custom_model_name is required when use_custom_model is True")
        
        if not self.custom_vector_dimension:
            raise ValueError("custom_vector_dimension is required when use_custom_model is True")
        
        # Parse additional files from comma-separated string
        additional_files = None
        if self.custom_additional_files:
            additional_files = [f.strip() for f in self.custom_additional_files.split(",")]
        
        return CustomModelSettings(
            model_name=self.custom_model_name,
            hf_model_id=self.custom_hf_model_id,
            model_url=self.custom_model_url,
            pooling_type=self.custom_pooling_type,
            normalization=self.custom_normalization,
            vector_dimension=self.custom_vector_dimension,
            model_file=self.custom_model_file,
            additional_files=additional_files,
        )


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
