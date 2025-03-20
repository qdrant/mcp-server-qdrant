from enum import Enum, auto
from typing import Dict, Any, List, Union, Optional


class EmbeddingProviderType(str, Enum):
    """
    Enumeration of supported embedding provider types.
    Using string enum for easier serialization/deserialization.
    """
    FASTEMBED = "fastembed"
    # Add other providers as they are implemented
    # OPENAI = "openai"
    # COHERE = "cohere"
    # etc.


class EmbeddingVector(List[float]):
    """Type alias for embedding vectors."""
    pass


class ModelMetadata(Dict[str, Any]):
    """Type alias for model metadata."""
    pass
