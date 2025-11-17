from enum import Enum


class EmbeddingProviderType(Enum):
    FASTEMBED = "fastembed"
    MODEL2VEC = "model2vec"
    OAI_COMPAT = "oai_compat"
    FASTEMBED_UNNAMED = "fastembed_unnamed"
