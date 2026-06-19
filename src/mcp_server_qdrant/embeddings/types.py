from enum import Enum


class EmbeddingProviderType(str, Enum):
    FASTEMBED = "fastembed"
    OPENAI = "openai"
