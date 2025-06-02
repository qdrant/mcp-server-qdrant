from enum import Enum


class EmbeddingProviderType(Enum):
    FASTEMBED = "fastembed"
    GOOGLE_GENAI = "google_genai"
