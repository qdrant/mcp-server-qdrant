from typing import Any, Optional, Union

from pydantic import BaseModel, Field

from qdrant_client.conversions.common_types import SparseVector
from qdrant_client.http import models

try:
    from fastembed import (
        SparseTextEmbedding,
        TextEmbedding,
        LateInteractionTextEmbedding,
        ImageEmbedding,
        LateInteractionMultimodalEmbedding,
    )
    from fastembed.text.multitask_embedding import JinaEmbeddingV3 as _MultitaskTextEmbedding
    from fastembed.common import OnnxProvider, ImageInput
except ImportError:
    TextEmbedding = None
    SparseTextEmbedding = None
    OnnxProvider = None
    LateInteractionTextEmbedding = None
    LateInteractionMultimodalEmbedding = None
    ImageEmbedding = None
    ImageInput = None


SUPPORTED_EMBEDDING_MODELS: dict[str, tuple[int, models.Distance]] = (
    {
        model["model"]: (model["dim"], models.Distance.COSINE)
        for model in TextEmbedding.list_supported_models()
    }
    if TextEmbedding
    else {}
)

SUPPORTED_SPARSE_EMBEDDING_MODELS: dict[str, dict[str, Any]] = (
    {model["model"]: model for model in SparseTextEmbedding.list_supported_models()}
    if SparseTextEmbedding
    else {}
)

IDF_EMBEDDING_MODELS: set[str] = (
    {
        model_config["model"]
        for model_config in SparseTextEmbedding.list_supported_models()
        if model_config.get("requires_idf", None)
    }
    if SparseTextEmbedding
    else set()
)

_LATE_INTERACTION_EMBEDDING_MODELS: dict[str, tuple[int, models.Distance]] = (
    {
        model["model"]: (model["dim"], models.Distance.COSINE)
        for model in LateInteractionTextEmbedding.list_supported_models()
    }
    if LateInteractionTextEmbedding
    else {}
)

_IMAGE_EMBEDDING_MODELS: dict[str, tuple[int, models.Distance]] = (
    {
        model["model"]: (model["dim"], models.Distance.COSINE)
        for model in ImageEmbedding.list_supported_models()
    }
    if ImageEmbedding
    else {}
)

_LATE_INTERACTION_MULTIMODAL_EMBEDDING_MODELS: dict[str, tuple[int, models.Distance]] = (
    {
        model["model"]: (model["dim"], models.Distance.COSINE)
        for model in LateInteractionMultimodalEmbedding.list_supported_models()
    }
    if LateInteractionMultimodalEmbedding
    else {}
)


class QueryResponse(BaseModel, extra="forbid"):  # type: ignore
    id: Union[str, int]
    embedding: Optional[list[float]]
    sparse_embedding: Optional[SparseVector] = Field(default=None)
    metadata: dict[str, Any]
    document: str
    score: float
