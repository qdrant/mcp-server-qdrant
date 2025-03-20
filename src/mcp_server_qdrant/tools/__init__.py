"""
Custom tools for Qdrant MCP server.

This package contains various custom tools for extending the functionality
of the Qdrant MCP server:

- Search tools: Natural language, hybrid, and multi-vector search
- Collection management tools: Creating and migrating collections
- Data processing tools: Batch embedding and text chunking
- Advanced query tools: Semantic routing and query decomposition
- Analytics tools: Collection analysis and query benchmarking
- Document processing tools: Document indexing and PDF processing
"""

# Search tools
from .search.nlq import natural_language_query
from .search.hybrid import hybrid_search
from .search.multi_vector import multi_vector_search

# Collection management tools
from .collection_mgmt.create_collection import create_collection
from .collection_mgmt.migrate_collection import migrate_collection

# Data processing tools
from .data_processing.batch_embed import batch_embed
from .data_processing.chunk_and_process import chunk_and_process

# Advanced query tools
from .advanced_query.semantic_router import semantic_router
from .advanced_query.decompose_query import decompose_query

# Analytics tools
from .analytics.analyze_collection import analyze_collection
from .analytics.benchmark_query import benchmark_query

# Document processing tools
from .document_processing.index_document import index_document
from .document_processing.process_pdf import process_pdf

__all__ = [
    # Search tools
    "natural_language_query",
    "hybrid_search",
    "multi_vector_search",
    
    # Collection management tools
    "create_collection",
    "migrate_collection",
    
    # Data processing tools
    "batch_embed",
    "chunk_and_process",
    
    # Advanced query tools
    "semantic_router",
    "decompose_query",
    
    # Analytics tools
    "analyze_collection",
    "benchmark_query",
    
    # Document processing tools
    "index_document",
    "process_pdf",
]
