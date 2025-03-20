

### Multilingual Search Tool

```python
# tools/multilingual/multilingual_search.py
from typing import Dict, Any, List, Optional

from mcp.server.fastmcp import Context
from mcp_server_qdrant.tools.search.nlq import natural_language_query
from mcp_server_qdrant.tools.multilingual.translate import translate_text
from mcp_server_qdrant.tools.multilingual.language_detection import detect_language

async def multilingual_search(
    ctx: Context,
    query: str,
    collection: str,
    target_language: Optional[str] = None,
    translate_results: bool = True,
    limit: int = 10,
    filter: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Perform cross-language search and translate results.
    
    Parameters:
    -----------
    ctx : Context
        MCP request context
    query : str
        Query in any language
    collection : str
        Collection to search
    target_language : str, optional
        Target language for results (default: same as query)
    translate_results : bool, default=True
        Whether to translate search results
    limit : int, default=10
        Maximum number of results to return
    filter : Dict[str, Any], optional
        Filter to apply to search
        
    Returns:
    --------
    Dict[str, Any]
        Search results with translations
    """
    await ctx.debug(f"Performing multilingual search for query: {query}")
    
    try:
        # Detect query language
        query_lang_result = await detect_language(ctx, query)
        query_language = query_lang_result.get("detected_language", "en")
        
        # If target language not specified, use query language
        if not target_language:
            target_language = query_language
        
        # Translate query to English for better embedding
        if query_language != "en":
            translation_result = await translate_text(
                ctx=ctx,
                text=query,
                source_language=query_language,
                target_language="en"
            )
            
            english_query = translation_result.get("translated_text", query)
            await ctx.debug(f"Translated query to English: {english_query}")
        else:
            english_query = query
        
        # Search with English query
        search_results = await natural_language_query(
            ctx=ctx,
            query=english_query,
            collection=collection,
            limit=limit,
            filter=filter
        )
        
        # Translate results if needed
        if translate_results and target_language != "en":
            for result in search_results:
                document = result.get("document", "")
                
                if document:
                    # Detect document language
                    doc_lang_result = await detect_language(ctx, document[:1000])
                    doc_language = doc_lang_result.get("detected_language", "en")
                    
                    # Translate if needed
                    if doc_language != target_language:
                        translation_result = await translate_text(
                            ctx=ctx,
                            text=document,
                            source_language=doc_language,
                            target_language=target_language
                        )
                        
                        # Add translation to result
                        result["original_document"] = document
                        result["document"] = translation_result.get("translated_text", document)
                        result["original_language"] = doc_language
        
        return {
            "status": "success",
            "query": query,
            "query_language": query_language,
            "target_language": target_language,
            "translated_query": english_query if query_language != "en" else None,
            "results": search_results
        }
    
    except Exception as e:
        await ctx.debug(f"Error in multilingual search: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to perform multilingual search: {str(e)}"
        }
```

### Integration with MCP Server

```python
@mcp.tool(name="detect-language", description=tool_settings.detect_language_description)
async def detect_language_tool(ctx: Context, **kwargs) -> Dict[str, Any]:
    return await detect_language(ctx, **kwargs)

@mcp.tool(name="translate-text", description=tool_settings.translate_text_description)
async def translate_text_tool(ctx: Context, **kwargs) -> Dict[str, Any]:
    return await translate_text(ctx, **kwargs)

@mcp.tool(name="multilingual-search", description=tool_settings.multilingual_search_description)
async def multilingual_search_tool(ctx: Context, **kwargs) -> Dict[str, Any]:
    return await multilingual_search(ctx, **kwargs)
```

### Dependencies

Add to `setup.py` or `pyproject.toml`:
```
langdetect>=1.0.9
deep-translator>=1.10.0
```

## Implementation Strategy

To implement these extensions effectively, follow this strategy:

### 1. Prioritize Extensions

Implement extensions in this order based on value and complexity:

1. **Vector Space Visualization** - High impact, moderate complexity
2. **Automated Metadata Extraction** - High impact, moderate complexity
3. **Document Versioning** - Medium impact, medium complexity
4. **Multilingual Support** - High impact, high complexity
5. **Semantic Clustering** - Medium impact, high complexity
6. **Advanced Embedding Models** - High impact, high complexity
7. **Real-time Connectors** - Medium impact, medium complexity

### 2. Dependency Management

Organize dependencies in `setup.py` or `pyproject.toml` with optional extras:

```python
extras_require={
    "visualization": [
        "umap-learn>=0.5.3",
        "scikit-learn>=1.0.2",
        "plotly>=5.10.0"
    ],
    "clustering": [
        "scikit-learn>=1.0.2",
        "hdbscan>=0.8.29",
        "nltk>=3.7"
    ],
    "multilingual": [
        "langdetect>=1.0.9",
        "deep-translator>=1.10.0"
    ],
    "connectors": [
        "feedparser>=6.0.0"
    ],
    "all": [
        # All dependencies combined
    ]
}
```

### 3. Testing Strategy

Create specific tests for each extension:

```python
# tests/unit/test_visualization.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from mcp_server_qdrant.tools.visualization.vector_projection import visualize_vectors

@pytest.mark.unit
async def test_visualize_vectors(mock_context):
    # Test the visualization tool
    pass

# tests/integration/test_multilingual.py
@pytest.mark.integration
async def test_multilingual_search(test_qdrant_connector):
    # Test multilingual search with actual translations
    pass
```

### 4. Documentation Updates

For each extension, update README.md with:

- Feature description
- Example use cases
- Configuration options
- Required dependencies

### 5. Phased Rollout

1. **Phase 1**: Core extensions (Visualization, Metadata Extraction)
2. **Phase 2**: Intermediate extensions (Versioning, Clustering)
3. **Phase 3**: Advanced extensions (Multilingual, Embedding Models)
4. **Phase 4**: Specialized extensions (Real-time Connectors)

By following this implementation strategy, you'll enhance your Qdrant MCP server with powerful capabilities while maintaining maintainability and flexibility.
# Extending Qdrant MCP Server Functionality

This guide outlines advanced extensions for the Qdrant MCP server beyond the already-planned web crawling integration, focusing on capabilities that enhance vector search, data processing, and user experience.

## Table of Contents

- [Introduction](#introduction)
- [Vector Space Visualization Tools](#vector-space-visualization-tools)
- [Document Versioning and Change Tracking](#document-versioning-and-change-tracking)
- [Advanced Embedding Models and Fusion](#advanced-embedding-models-and-fusion)
- [Automated Metadata Extraction](#automated-metadata-extraction)
- [Semantic Clustering and Topic Modeling](#semantic-clustering-and-topic-modeling)
- [Real-time Data Connectors](#real-time-data-connectors)
- [Multilingual Support](#multilingual-support)
- [Implementation Strategy](#implementation-strategy)

## Introduction

With web crawling integration already planned, we can focus on more advanced extensions that transform your Qdrant MCP server into a comprehensive vector database platform. These extensions provide additional capabilities for analysis, visualization, and sophisticated data handling.

## Vector Space Visualization Tools

### Overview

Create tools to visualize the vector space, enabling users to understand the relationships between vectors and improve their search strategies.

### Implementation

```python
# tools/visualization/vector_projection.py
import numpy as np
from typing import Dict, Any, List, Optional
import umap
import plotly.graph_objects as go
import plotly.io as pio
import json

from mcp.server.fastmcp import Context

async def visualize_vectors(
    ctx: Context,
    collection: str,
    algorithm: str = "umap",  # "umap", "tsne", or "pca"
    dimensions: int = 2,      # 2D or 3D visualization
    sample_size: int = 1000,  # Maximum vectors to sample
    query: Optional[str] = None,  # Optional query to highlight
    group_by_metadata: Optional[str] = None,  # Metadata field for coloring
    point_size: int = 5,
    include_html: bool = True,  # Include HTML visualization
) -> Dict[str, Any]:
    """
    Generate a 2D/3D projection of vectors in a collection for visualization.
    
    Parameters:
    -----------
    ctx : Context
        MCP request context
    collection : str
        Collection to visualize
    algorithm : str, default="umap"
        Dimensionality reduction algorithm ("umap", "tsne", "pca")
    dimensions : int, default=2
        Output dimensions (2 or 3)
    sample_size : int, default=1000
        Maximum number of points to sample
    query : str, optional
        Optional query to highlight in the visualization
    group_by_metadata : str, optional
        Metadata field to use for coloring points
    point_size : int, default=5
        Size of points in the visualization
    include_html : bool, default=True
        Whether to include HTML visualization in the response
        
    Returns:
    --------
    Dict[str, Any]
        Visualization data and optionally HTML
    """
    await ctx.debug(f"Generating {dimensions}D vector visualization for collection: {collection}")
    
    # Get Qdrant client
    qdrant_connector = ctx.request_context.lifespan_context["qdrant_connector"]
    client = qdrant_connector._client
    embedding_provider = qdrant_connector._embedding_provider
    
    # Sample points from collection
    try:
        # Get points with vectors and payload
        response = await client.scroll(
            collection_name=collection,
            limit=sample_size,
            with_vectors=True,
            with_payload=True
        )
        
        points = response.points
        await ctx.debug(f"Retrieved {len(points)} points for visualization")
        
        if not points:
            return {
                "status": "error",
                "message": "No points found in collection or collection is empty"
            }
        
        # Extract vectors, ids, and payloads
        vectors = []
        point_ids = []
        payloads = []
        
        for point in points:
            # Handle different vector formats
            if hasattr(point, "vector") and point.vector:
                if isinstance(point.vector, dict):
                    # Handle named vectors - use the first one
                    vector_name = next(iter(point.vector))
                    vector = point.vector[vector_name]
                else:
                    vector = point.vector
                
                vectors.append(vector)
                point_ids.append(str(point.id))
                payloads.append(point.payload or {})
        
        if not vectors:
            return {
                "status": "error",
                "message": "No valid vectors found in the sampled points"
            }
        
        # If query provided, add it to the visualization
        query_vector = None
        if query:
            query_vector = await embedding_provider.embed_query(query)
            vectors.append(query_vector)
            point_ids.append("query")
            payloads.append({"is_query": True})
        
        # Convert to numpy array
        vectors_array = np.array(vectors)
        
        # Apply dimensionality reduction
        if algorithm.lower() == "umap":
            import umap
            reducer = umap.UMAP(n_components=dimensions, random_state=42)
            embedding = reducer.fit_transform(vectors_array)
        elif algorithm.lower() == "tsne":
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=dimensions, random_state=42)
            embedding = reducer.fit_transform(vectors_array)
        elif algorithm.lower() == "pca":
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=dimensions)
            embedding = reducer.fit_transform(vectors_array)
        else:
            return {
                "status": "error",
                "message": f"Unknown algorithm: {algorithm}. Supported: umap, tsne, pca"
            }
        
        # Prepare visualization data
        viz_data = []
        for i, (id, payload, coords) in enumerate(zip(point_ids, payloads, embedding)):
            point_data = {
                "id": id,
                "coordinates": coords.tolist(),
            }
            
            # Add payload data
            if group_by_metadata and group_by_metadata in payload:
                point_data["group"] = str(payload[group_by_metadata])
            elif "is_query" in payload and payload["is_query"]:
                point_data["group"] = "query"
            else:
                point_data["group"] = "default"
            
            # Add payload excerpt
            point_data["payload"] = {k: v for k, v in payload.items() if isinstance(v, (str, int, float, bool))}
            
            viz_data.append(point_data)
        
        # Create interactive visualization if requested
        html_output = None
        if include_html:
            # Group data by group value
            groups = {}
            for point in viz_data:
                group = point["group"]
                if group not in groups:
                    groups[group] = {
                        "ids": [],
                        "x": [],
                        "y": [],
                        "z": [] if dimensions == 3 else None,
                        "text": []
                    }
                
                groups[group]["ids"].append(point["id"])
                groups[group]["x"].append(point["coordinates"][0])
                groups[group]["y"].append(point["coordinates"][1])
                if dimensions == 3:
                    groups[group]["z"].append(point["coordinates"][2])
                
                # Create hover text
                hover_text = f"ID: {point['id']}<br>"
                for k, v in point["payload"].items():
                    if isinstance(v, str) and len(v) > 50:
                        v = v[:50] + "..."
                    hover_text += f"{k}: {v}<br>"
                
                groups[group]["text"].append(hover_text)
            
            # Create plotly figure
            fig = go.Figure()
            
            for group_name, group_data in groups.items():
                if dimensions == 3:
                    fig.add_trace(go.Scatter3d(
                        x=group_data["x"],
                        y=group_data["y"],
                        z=group_data["z"],
                        mode="markers",
                        marker=dict(
                            size=12 if group_name == "query" else point_size,
                            opacity=0.8
                        ),
                        text=group_data["text"],
                        hoverinfo="text",
                        name=group_name
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=group_data["x"],
                        y=group_data["y"],
                        mode="markers",
                        marker=dict(
                            size=12 if group_name == "query" else point_size,
                            opacity=0.8
                        ),
                        text=group_data["text"],
                        hoverinfo="text",
                        name=group_name
                    ))
            
            # Update layout
            fig.update_layout(
                title=f"{dimensions}D Vector Space Visualization ({algorithm.upper()})",
                legend_title="Groups",
                height=800,
                template="plotly_white"
            )
            
            if dimensions == 2:
                fig.update_layout(
                    xaxis_title="Dimension 1",
                    yaxis_title="Dimension 2"
                )
            else:
                fig.update_layout(
                    scene=dict(
                        xaxis_title="Dimension 1",
                        yaxis_title="Dimension 2",
                        zaxis_title="Dimension 3"
                    )
                )
            
            # Convert to HTML
            html_output = pio.to_html(fig, full_html=True, include_plotlyjs='cdn')
        
        # Return visualization data
        return {
            "status": "success",
            "collection": collection,
            "algorithm": algorithm,
            "dimensions": dimensions,
            "points_count": len(viz_data),
            "visualization_data": viz_data,
            "html": html_output if include_html else None
        }
    
    except Exception as e:
        await ctx.debug(f"Error generating visualization: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to generate visualization: {str(e)}"
        }
```

### Integration with MCP Server

To surface these visualizations to clients, we'll need to add a tool registration:

```python
@mcp.tool(name="visualize-vectors", description=tool_settings.visualize_vectors_description)
async def visualize_vectors_tool(ctx: Context, **kwargs) -> Dict[str, Any]:
    return await visualize_vectors(ctx, **kwargs)
```

### Dependencies

Add to `setup.py` or `pyproject.toml`:
```
umap-learn>=0.5.3
scikit-learn>=1.0.2
plotly>=5.10.0
```

## Document Versioning and Change Tracking

### Overview

Implement version control for documents in Qdrant, allowing users to track changes over time and maintain a history of indexed content.

### Implementation

```python
# tools/versioning/document_version.py
from datetime import datetime
from typing import Dict, Any, Optional, List
import uuid
import difflib

from mcp.server.fastmcp import Context
from mcp_server_qdrant.tools.data_processing.chunk_and_process import chunk_and_process

async def version_document(
    ctx: Context,
    document_id: str,
    new_content: str,
    collection: str,
    version_collection: Optional[str] = None,
    track_changes: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> Dict[str, Any]:
    """
    Update a document while maintaining version history.
    
    Parameters:
    -----------
    ctx : Context
        MCP request context
    document_id : str
        Unique identifier for the document
    new_content : str
        Updated content for the document
    collection : str
        Collection containing the document
    version_collection : str, optional
        Collection to store version history (defaults to "{collection}_versions")
    track_changes : bool, default=True
        Whether to compute and store differences between versions
    metadata : Dict[str, Any], optional
        Additional metadata for the new version
    chunk_size : int, default=1000
        Size of chunks for processing
    chunk_overlap : int, default=200
        Overlap between chunks
        
    Returns:
    --------
    Dict[str, Any]
        Version information
    """
    await ctx.debug(f"Versioning document: {document_id}")
    
    # Get Qdrant client
    qdrant_connector = ctx.request_context.lifespan_context["qdrant_connector"]
    client = qdrant_connector._client
    
    # If version collection not specified, use default naming
    if not version_collection:
        version_collection = f"{collection}_versions"
    
    # Initialize metadata if not provided
    if metadata is None:
        metadata = {}
    
    try:
        # Check if version collection exists, create if not
        try:
            await client.get_collection(version_collection)
        except Exception:
            # Get vector configuration from main collection
            collection_info = await client.get_collection(collection)
            vectors_config = collection_info.config.params.vectors
            
            # Create version collection with same configuration
            await client.create_collection(
                collection_name=version_collection,
                vectors_config=vectors_config
            )
            await ctx.debug(f"Created version collection: {version_collection}")
        
        # Retrieve current document chunks
        filter_query = {
            "must": [
                {
                    "key": "metadata.document_id",
                    "match": {"value": document_id}
                }
            ]
        }
        
        current_chunks_response = await client.scroll(
            collection_name=collection,
            filter=filter_query,
            limit=100,  # Adjust as needed for larger documents
            with_payload=True,
            with_vectors=False
        )
        
        current_chunks = current_chunks_response.points
        
        # Extract current content if document exists
        current_content = ""
        current_version = 0
        current_metadata = {}
        
        if current_chunks:
            # Sort chunks by index if available
            try:
                current_chunks.sort(key=lambda x: x.payload.get("chunk_index", 0))
            except Exception:
                pass
            
            # Concatenate content from chunks
            current_content = " ".join(
                chunk.payload.get("text", "") for chunk in current_chunks if "text" in chunk.payload
            )
            
            # Get current version number
            for chunk in current_chunks:
                chunk_metadata = chunk.payload.get("metadata", {})
                if "version" in chunk_metadata:
                    current_version = max(current_version, chunk_metadata["version"])
                current_metadata.update({k: v for k, v in chunk_metadata.items() 
                                      if k not in ["document_id", "version", "created_at"]})
        
        # Increment version
        new_version = current_version + 1
        
        # Generate diff if tracking changes
        diff_text = None
        if track_changes and current_content:
            diff = difflib.unified_diff(
                current_content.splitlines(),
                new_content.splitlines(),
                fromfile=f"version_{current_version}",
                tofile=f"version_{new_version}",
                lineterm=""
            )
            diff_text = "\n".join(diff)
        
        # Create version metadata
        version_metadata = {
            "document_id": document_id,
            "version": new_version,
            "previous_version": current_version if current_version > 0 else None,
            "created_at": datetime.now().isoformat(),
            "has_diff": diff_text is not None,
            **current_metadata,  # Include previous metadata
            **metadata  # Override with new metadata
        }
        
        # Store version information in version collection
        version_id = f"{document_id}_v{new_version}"
        
        # Process new content
        chunks_result = await chunk_and_process(
            ctx=ctx,
            text=new_content,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            collection=None,  # Don't store directly
            metadata=version_metadata
        )
        
        processed_chunks = chunks_result.get("chunks", [])
        
        # If we have a diff, store it with the first chunk
        if diff_text and processed_chunks:
            processed_chunks[0]["payload"]["diff"] = diff_text
        
        # Store version in version collection
        if processed_chunks:
            await client.upsert(
                collection_name=version_collection,
                points=processed_chunks
            )
        
        # Delete current document chunks
        if current_chunks:
            chunk_ids = [chunk.id for chunk in current_chunks]
            await client.delete(
                collection_name=collection,
                points_selector={"points": chunk_ids}
            )
        
        # Update document chunks with new content
        for chunk in processed_chunks:
            # Update payload to include document_id
            chunk["payload"]["metadata"]["document_id"] = document_id
        
        # Store updated chunks in main collection
        await client.upsert(
            collection_name=collection,
            points=processed_chunks
        )
        
        return {
            "status": "success",
            "document_id": document_id,
            "version": new_version,
            "previous_version": current_version if current_version > 0 else None,
            "has_diff": diff_text is not None,
            "chunks_count": len(processed_chunks),
            "version_collection": version_collection
        }
    
    except Exception as e:
        await ctx.debug(f"Error versioning document: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to version document: {str(e)}"
        }
```

### Get Document History Tool

```python
# tools/versioning/document_history.py
from typing import Dict, Any, List, Optional
from datetime import datetime

from mcp.server.fastmcp import Context

async def get_document_history(
    ctx: Context,
    document_id: str,
    version_collection: str,
    include_diffs: bool = False,
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Retrieve version history for a document.
    
    Parameters:
    -----------
    ctx : Context
        MCP request context
    document_id : str
        Document ID to retrieve history for
    version_collection : str
        Collection containing version history
    include_diffs : bool, default=False
        Whether to include content diffs in the response
    limit : int, default=10
        Maximum number of versions to retrieve
        
    Returns:
    --------
    Dict[str, Any]
        Document version history
    """
    await ctx.debug(f"Retrieving history for document: {document_id}")
    
    # Get Qdrant client
    qdrant_connector = ctx.request_context.lifespan_context["qdrant_connector"]
    client = qdrant_connector._client
    
    try:
        # Query versions
        filter_query = {
            "must": [
                {
                    "key": "metadata.document_id",
                    "match": {"value": document_id}
                }
            ]
        }
        
        # Get first chunk of each version (containing metadata and diff)
        versions_response = await client.scroll(
            collection_name=version_collection,
            filter=filter_query,
            limit=limit * 10,  # Get more chunks to find all versions
            with_payload=True,
            with_vectors=False
        )
        
        versions_chunks = versions_response.points
        
        # Group chunks by version
        versions_map = {}
        for chunk in versions_chunks:
            metadata = chunk.payload.get("metadata", {})
            version = metadata.get("version")
            
            if version is not None and version not in versions_map:
                versions_map[version] = {
                    "version": version,
                    "created_at": metadata.get("created_at"),
                    "previous_version": metadata.get("previous_version"),
                    "has_diff": metadata.get("has_diff", False),
                    "diff": chunk.payload.get("diff") if include_diffs and "diff" in chunk.payload else None,
                    "metadata": {k: v for k, v in metadata.items() 
                               if k not in ["document_id", "version", "previous_version", "created_at", "has_diff"]}
                }
        
        # Convert to sorted list
        versions = list(versions_map.values())
        versions.sort(key=lambda x: x["version"], reverse=True)
        
        # Limit number of versions
        versions = versions[:limit]
        
        return {
            "status": "success",
            "document_id": document_id,
            "versions_count": len(versions),
            "versions": versions
        }
    
    except Exception as e:
        await ctx.debug(f"Error retrieving document history: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to retrieve document history: {str(e)}"
        }
```

### Integration with MCP Server

```python
@mcp.tool(name="version-document", description=tool_settings.version_document_description)
async def version_document_tool(ctx: Context, **kwargs) -> Dict[str, Any]:
    return await version_document(ctx, **kwargs)

@mcp.tool(name="document-history", description=tool_settings.document_history_description)
async def document_history_tool(ctx: Context, **kwargs) -> Dict[str, Any]:
    return await get_document_history(ctx, **kwargs)
```

## Advanced Embedding Models and Fusion

### Overview

Support for multiple embedding models and sophisticated fusion techniques to enhance search quality.

### Implementation

```python
# tools/embeddings/fusion.py
from typing import Dict, Any, List, Optional
import numpy as np

from mcp.server.fastmcp import Context

async def fusion_search(
    ctx: Context,
    query: str,
    collection: str,
    models: List[str],
    fusion_method: str = "rrf",  # "rrf", "linear", "max", "min"
    weights: Optional[List[float]] = None,
    limit: int = 10,
    filter: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Search using multiple embedding models with fusion of results.
    
    Parameters:
    -----------
    ctx : Context
        MCP request context
    query : str
        Search query
    collection : str
        Collection to search
    models : List[str]
        List of embedding models to use
    fusion_method : str, default="rrf"
        Method for fusing results: 
        - "rrf": Reciprocal Rank Fusion
        - "linear": Weighted linear combination
        - "max": Maximum score across models
        - "min": Minimum score across models
    weights : List[float], optional
        Weights for each model (must match length of models)
    limit : int, default=10
        Maximum number of results to return
    filter : Dict[str, Any], optional
        Filter to apply to search
        
    Returns:
    --------
    Dict[str, Any]
        Fused search results
    """
    await ctx.debug(f"Fusion search with {len(models)} models using {fusion_method}")
    
    # Get dependencies
    qdrant_connector = ctx.request_context.lifespan_context["qdrant_connector"]
    client = qdrant_connector._client
    default_embedding_provider = qdrant_connector._embedding_provider
    
    # Validate models
    available_models = await default_embedding_provider.list_available_models()
    for model in models:
        if model not in available_models:
            return {
                "status": "error",
                "message": f"Model '{model}' is not available. Available models: {available_models}"
            }
    
    # Validate weights if provided
    if weights:
        if len(weights) != len(models):
            return {
                "status": "error",
                "message": f"Number of weights ({len(weights)}) must match number of models ({len(models)})"
            }
    else:
        # Default to equal weights
        weights = [1.0 / len(models)] * len(models)
    
    try:
        # Generate embeddings for each model
        model_embeddings = []
        for model in models:
            embedding = await default_embedding_provider.embed_query(query, model=model)
            model_embeddings.append(embedding)
        
        all_results = {}  # Map of point_id -> result info
        
        # Perform searches with each model
        for i, (model, embedding) in enumerate(zip(models, model_embeddings)):
            search_results = await client.search(
                collection_name=collection,
                query_vector=embedding,
                limit=limit * 3,  # Get more results for better fusion
                filter=filter,
                with_payload=True
            )
            
            # Process results for this model
            for rank, result in enumerate(search_results):
                point_id = str(result.id)
                
                if point_id not in all_results:
                    all_results[point_id] = {
                        "id": point_id,
                        "payload": result.payload,
                        "model_scores": [0.0] * len(models),
                        "model_ranks": [float('inf')] * len(models),
                        "final_score": 0.0
                    }
                
                # Store score and rank for this model
                all_results[point_id]["model_scores"][i] = float(result.score)
                all_results[point_id]["model_ranks"][i] = rank + 1  # 1-based rank
        
        # Apply fusion method
        results_list = list(all_results.values())
        
        if fusion_method.lower() == "rrf":
            # Reciprocal Rank Fusion
            k = 60  # RRF constant - prevents tiny contributions from very low ranks
            
            for result in results_list:
                rrf_score = sum(weights[i] * (1.0 / (k + rank)) 
                             for i, (rank, weight) in enumerate(zip(result["model_ranks"], weights)))
                result["final_score"] = rrf_score
        
        elif fusion_method.lower() == "linear":
            # Weighted linear combination of scores
            for result in results_list:
                linear_score = sum(score * weight for score, weight in zip(result["model_scores"], weights))
                result["final_score"] = linear_score
        
        elif fusion_method.lower() == "max":
            # Maximum score across models
            for result in results_list:
                # Apply weights before taking max
                weighted_scores = [score * weight for score, weight in zip(result["model_scores"], weights)]
                result["final_score"] = max(weighted_scores)
        
        elif fusion_method.lower() == "min":
            # Minimum score across models (good for ensuring all models agree)
            for result in results_list:
                # Apply weights before taking min
                weighted_scores = [score * weight for score, weight in zip(result["model_scores"], weights)]
                result["final_score"] = min(weighted_scores)
        
        else:
            return {
                "status": "error",
                "message": f"Unknown fusion method: {fusion_method}. Supported: rrf, linear, max, min"
            }
        
        # Sort by final score and limit results
        results_list.sort(key=lambda x: x["final_score"], reverse=True)
        top_results = results_list[:limit]
        
        # Format results
        formatted_results = []
        for result in top_results:
            formatted_result = {
                "id": result["id"],
                "score": result["final_score"],
                "document": result["payload"].get("document", result["payload"].get("text", "")),
                "metadata": result["payload"].get("metadata", {}),
                "model_scores": {model: score for model, score in zip(models, result["model_scores"])},
            }
            formatted_results.append(formatted_result)
        
        return {
            "status": "success",
            "query": query,
            "fusion_method": fusion_method,
            "models": models,
            "weights": weights,
            "count": len(formatted_results),
            "results": formatted_results
        }
    
    except Exception as e:
        await ctx.debug(f"Error in fusion search: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to perform fusion search: {str(e)}"
        }
```

### Integration with MCP Server

```python
@mcp.tool(name="fusion-search", description=tool_settings.fusion_search_description)
async def fusion_search_tool(ctx: Context, **kwargs) -> Dict[str, Any]:
    return await fusion_search(ctx, **kwargs)
```

## Automated Metadata Extraction

### Overview

Extract structured metadata from documents automatically to improve search filtering and faceting.

### Implementation

```python
# tools/data_processing/metadata_extraction.py
from typing import Dict, Any, List, Optional
import re
import datetime

from mcp.server.fastmcp import Context
from mcp_server_qdrant.tools.utils import extract_text

async def extract_metadata(
    ctx: Context,
    text: str,
    extractors: Optional[List[str]] = None,
    use_llm: bool = False,
    custom_patterns: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Extract structured metadata from text.
    
    Parameters:
    -----------
    ctx : Context
        MCP request context
    text : str
        Text to extract metadata from
    extractors : List[str], optional
        List of metadata types to extract (default: all available)
    use_llm : bool, default=False
        Whether to use an LLM for advanced extraction
    custom_patterns : Dict[str, str], optional
        Dictionary of custom regex patterns for extraction
        
    Returns:
    --------
    Dict[str, Any]
        Extracted metadata
    """
    await ctx.debug(f"Extracting metadata from text ({len(text)} chars)")
    
    # Define available extractors
    available_extractors = {
        "dates": extract_dates,
        "emails": extract_emails,
        "urls": extract_urls,
        "people": extract_people,
        "organizations": extract_organizations,
        "locations": extract_locations,
        "phone_numbers": extract_phone_numbers,
        "categories": categorize_content
    }
    
    # Determine which extractors to use
    if not extractors:
        extractors = list(available_extractors.keys())
    else:
        # Validate requested extractors
        invalid_extractors = [e for e in extractors if e not in available_extractors]
        if invalid_extractors:
            return {
                "status": "error",
                "message": f"Invalid extractors: {invalid_extractors}. Available: {list(available_extractors.keys())}"
            }
    
    try:
        # Initialize metadata
        metadata = {}
        
        # Apply extractors
        for extractor_name in extractors:
            extractor_func = available_extractors[extractor_name]
            
            # Call extractor function
            if extractor_name == "categories" and use_llm:
                # Special case for categorization with LLM
                result = await categorize_content_llm(ctx, text)
            else:
                result = extractor_func(text, custom_patterns.get(extractor_name) if custom_patterns else None)
            
            # Add to metadata
            metadata[extractor_name] = result
        
        # Use LLM for advanced extraction if requested
        if use_llm and "llm_extraction" not in metadata:
            llm_result = await extract_with_llm(ctx, text)
            metadata["llm_extraction"] = llm_result
        
        return {
            "status": "success",
            "metadata": metadata,
            "extractors_used": extractors,
            "used_llm": use_llm
        }
    
    except Exception as e:
        await ctx.debug(f"Error extracting metadata: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to extract metadata: {str(e)}"
        }

# Helper functions for metadata extraction

def extract_dates(text: str, custom_pattern: Optional[str] = None) -> List[Dict[str, Any]]:
    """Extract dates from text."""
    results = []
    
    # ISO dates (YYYY-MM-DD)
    iso_pattern = r'\b(\d{4}-\d{2}-\d{2})\b'
    
    # US dates (MM/DD/YYYY)
    us_pattern = r'\b(\d{1,2}/\d{1,2}/\d{4})\b'
    
    # European dates (DD.MM.YYYY)
    eu_pattern = r'\b(\d{1,2}\.\d{1,2}\.\d{4})\b'
    
    # Written dates (Month DD, YYYY)
    written_pattern = r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b'
    
    # Custom pattern
    if custom_pattern:
        patterns = [custom_pattern]
    else:
        patterns = [iso_pattern, us_pattern, eu_pattern, written_pattern]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            date_str = match.group(0)
            try:
                # For simplicity, we're not parsing the actual date objects here
                # In a real implementation, you would convert to a standard format
                results.append({
                    "date": date_str,
                    "position": match.start()
                })
            except Exception:
                pass
    
    return results

def extract_emails(text: str, custom_pattern: Optional[str] = None) -> List[str]:
    """Extract email addresses from text."""
    pattern = custom_pattern or r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return list(set(re.findall(pattern, text)))

def extract_urls(text: str, custom_pattern: Optional[str] = None) -> List[str]:
    """Extract URLs from text."""
    pattern = custom_pattern or r'https?://[^\s<>"]+|www\.[^\s<>"]+\b'
    return list(set(re.findall(pattern, text)))

def extract_phone_numbers(text: str, custom_pattern: Optional[str] = None) -> List[str]:
    """Extract phone numbers from text."""
    pattern = custom_pattern or r'\b(\+\d{1,2}\s?)?(\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}\b'
    return list(set(re.findall(pattern, text)))

def extract_people(text: str, custom_pattern: Optional[str] = None) -> List[str]:
    """
    Extract people names from text.
    This is a simple pattern-based approach and would be much improved with NER models.
    """
    # Very basic pattern - would be better with NLP
    pattern = custom_pattern or r'(?:[A-Z][a-z]+\s+){1,2}[A-Z][a-z]+'
    candidates = re.findall(pattern, text)
    
    # Filter out obvious non-names
    common_words = ["The", "This", "That", "These", "Those", "There", "Their", "They", "About"]
    filtered = [name for name in candidates if not name.startswith(tuple(common_words))]
    
    return list(set(filtered))

def extract_organizations(text: str, custom_pattern: Optional[str] = None) -> List[str]:
    """
    Extract organization names from text.
    This is a simple pattern-based approach and would be much improved with NER models.
    """
    # Simple pattern for organizations - would be better with NLP
    pattern = custom_pattern or r'(?:[A-Z][a-z]*\s+)*(?:Inc|LLC|Ltd|Corp|Corporation|Company|Co|GmbH|AG|SA)\b'
    return list(set(re.findall(pattern, text)))

def extract_locations(text: str, custom_pattern: Optional[str] = None) -> List[str]:
    """
    Extract location names from text.
    This is a simple pattern-based approach and would be much improved with NER models.
    """
    # Very limited approach - would be better with a gazetteer or NLP
    pattern = custom_pattern or r'(?:in|at|from|to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
    matches = re.finditer(pattern, text)
    locations = [match.group(1) for match in matches]
    return list(set(locations))

def categorize_content(text: str, custom_categories: Optional[str] = None) -> Dict[str, float]:
    """
    Categorize text content using keyword matching.
    Returns confidence scores for each category.
    """
    # Define categories and their keywords
    if custom_categories:
        # Parse custom categories if provided
        try:
            categories = json.loads(custom_categories)
        except Exception:
            categories = {
                "technology": ["computer", "software", "hardware", "internet", "digital", "tech", "AI", "data"],
                "business": ["company", "market", "finance", "economic", "industry", "profit", "revenue"],
                "science": ["research", "scientific", "experiment", "theory", "physics", "biology", "chemistry"],
                "health": ["medical", "health", "doctor", "patient", "hospital", "treatment", "disease"],
                "politics": ["government", "policy", "election", "political", "president", "vote", "law"]
            }
    else:
        categories = {
            "technology": ["computer", "software", "hardware", "internet", "digital", "tech", "AI", "data"],
            "business": ["company", "market", "finance", "economic", "industry", "profit", "revenue"],
            "science": ["research", "scientific", "experiment", "theory", "physics", "biology", "chemistry"],
            "health": ["medical", "health", "doctor", "patient", "hospital", "treatment", "disease"],
            "politics": ["government", "policy", "election", "political", "president", "vote", "law"]
        }
    
    # Normalize text
    text_lower = text.lower()
    
    # Calculate scores
    scores = {}
    for category, keywords in categories.items():
        count = sum(text_lower.count(keyword.lower()) for keyword in keywords)
        scores[category] = min(1.0, count / 10)  # Normalize score between 0 and 1
    
    return scores

async def categorize_content_llm(ctx: Context, text: str) -> Dict[str, float]:
    """
    Categorize text content using LLM.
    This would call an LLM service in a real implementation.
    """
    # Placeholder for LLM categorization
    # In a real implementation, this would call an actual LLM service
    return {
        "technology": 0.8,
        "business": 0.4,
        "science": 0.2,
        "health": 0.1,
        "politics": 0.1
    }

async def extract_with_llm(ctx: Context, text: str) -> Dict[str, Any]:
    """
    Extract structured metadata using an LLM.
    This would call an LLM service in a real implementation.
    """
    # Placeholder for LLM extraction
    # In a real implementation, this would call an actual LLM service
    return {
        "summary": "This is a placeholder summary",
        "key_insights": ["Insight 1", "Insight 2", "Insight 3"],
        "sentiment": "positive",
        "reading_time": len(text) // 1000,  # ~1 minute per 1000 chars
        "complexity_score": 0.5,
        "audience_level": "intermediate"
    }
```

### Integration with MCP Server

```python
@mcp.tool(name="extract-metadata", description=tool_settings.extract_metadata_description)
async def extract_metadata_tool(ctx: Context, **kwargs) -> Dict[str, Any]:
    return await extract_metadata(ctx, **kwargs)
```

## Semantic Clustering and Topic Modeling

### Overview

Implement tools for automatically organizing documents into clusters and extracting topics to provide insights into collections.

### Implementation

```python
# tools/analytics/semantic_clustering.py
from typing import Dict, Any, List, Optional
import numpy as np
from datetime import datetime

from mcp.server.fastmcp import Context

async def semantic_clustering(
    ctx: Context,
    collection: str,
    num_clusters: int = 5,
    algorithm: str = "kmeans",  # "kmeans", "hdbscan", "agglomerative"
    sample_size: int = 1000,
    filter: Optional[Dict[str, Any]] = None,
    extract_topics: bool = True,
    store_clusters: bool = False,
) -> Dict[str, Any]:
    """
    Perform semantic clustering on documents in a collection.
    
    Parameters:
    -----------
    ctx : Context
        MCP request context
    collection : str
        Collection to analyze
    num_clusters : int, default=5
        Number of clusters to create (for kmeans and agglomerative)
    algorithm : str, default="kmeans"
        Clustering algorithm to use
    sample_size : int, default=1000
        Maximum number of points to sample
    filter : Dict[str, Any], optional
        Filter to apply when sampling points
    extract_topics : bool, default=True
        Whether to extract topics for each cluster
    store_clusters : bool, default=False
        Whether to store cluster assignments back to the collection
        
    Returns:
    --------
    Dict[str, Any]
        Clustering results and statistics
    """
    await ctx.debug(f"Performing semantic clustering on collection: {collection}")
    
    # Get Qdrant client
    qdrant_connector = ctx.request_context.lifespan_context["qdrant_connector"]
    client = qdrant_connector._client
    
    try:
        # Sample points from collection
        response = await client.scroll(
            collection_name=collection,
            limit=sample_size,
            filter=filter,
            with_vectors=True,
            with_payload=True
        )
        
        points = response.points
        await ctx.debug(f"Retrieved {len(points)} points for clustering")
        
        if not points:
            return {
                "status": "error",
                "message": "No points found in collection or collection is empty"
            }
        
        # Extract vectors, ids, and payloads
        vectors = []
        point_ids = []
        payloads = []
        texts = []
        
        for point in points:
            # Handle different vector formats
            if hasattr(point, "vector") and point.vector:
                if isinstance(point.vector, dict):
                    # Handle named vectors - use the first one
                    vector_name = next(iter(point.vector))
                    vector = point.vector[vector_name]
                else:
                    vector = point.vector
                
                vectors.append(vector)
                point_ids.append(str(point.id))
                payloads.append(point.payload or {})
                
                # Extract text for topic modeling
                if extract_topics:
                    text = point.payload.get("text", point.payload.get("document", ""))
                    texts.append(text if isinstance(text, str) else "")
        
        if not vectors:
            return {
                "status": "error",
                "message": "No valid vectors found in the sampled points"
            }
        
        # Convert to numpy array
        vectors_array = np.array(vectors)
        
        # Perform clustering
        if algorithm.lower() == "kmeans":
            from sklearn.cluster import KMeans
            
            clustering = KMeans(n_clusters=num_clusters, random_state=42)
            cluster_labels = clustering.fit_predict(vectors_array)
            
            # Get cluster centers
            cluster_centers = clustering.cluster_centers_
            
        elif algorithm.lower() == "hdbscan":
            import hdbscan
            
            # HDBSCAN doesn't require num_clusters
            clustering = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
            cluster_labels = clustering.fit_predict(vectors_array)
            
            # For HDBSCAN, -1 represents noise points
            num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            
            # No direct cluster centers in HDBSCAN
            cluster_centers = None
            
        elif algorithm.lower() == "agglomerative":
            from sklearn.cluster import AgglomerativeClustering
            
            clustering = AgglomerativeClustering(n_clusters=num_clusters)
            cluster_labels = clustering.fit_predict(vectors_array)
            
            # No direct cluster centers in Agglomerative Clustering
            # We can compute the mean of points in each cluster
            cluster_centers = np.array([
                vectors_array[cluster_labels == i].mean(axis=0)
                for i in range(num_clusters)
            ])
            
        else:
            return {
                "status": "error",
                "message": f"Unknown algorithm: {algorithm}. Supported: kmeans, hdbscan, agglomerative"
            }
        
        # Group points by cluster
        clusters = {}
        for i, (point_id, payload, label) in enumerate(zip(point_ids, payloads, cluster_labels)):
            if label not in clusters:
                clusters[label] = {
                    "cluster_id": int(label),
                    "points": [],
                    "texts": []
                }
            
            # Add point to cluster
            clusters[label]["points"].append({
                "id": point_id,
                "payload": {k: v for k, v in payload.items() if isinstance(v, (str, int, float, bool))}
            })
            
            # Add text for topic modeling
            if extract_topics and i < len(texts):
                clusters[label]["texts"].append(texts[i])
        
        # Extract topics if requested
        if extract_topics and texts:
            try:
                for label, cluster in clusters.items():
                    # Only process clusters with texts
                    if cluster["texts"]:
                        # Extract topics using TF-IDF
                        cluster["topics"] = await extract_cluster_topics(
                            ctx, cluster["texts"], num_topics=5
                        )
            except Exception as e:
                await ctx.debug(f"Error extracting topics: {str(e)}")
                # Continue without topics
        
        # Store cluster assignments if requested
        if store_clusters:
            try:
                # Update payloads with cluster assignments
                for label, cluster in clusters.items():
                    for point in cluster["points"]:
                        await client.set_payload(
                            collection_name=collection,
                            payload={
                                "cluster": {
                                    "id": int(label),
                                    "algorithm": algorithm,
                                    "created_at": datetime.now().isoformat()
                                }
                            },
                            points=[point["id"]]
                        )
                
                await ctx.debug(f"Stored cluster assignments for {len(point_ids)} points")
            except Exception as e:
                await ctx.debug(f"Error storing cluster assignments: {str(e)}")
                # Continue without storing
        
        # Prepare final result
        result = {
            "status": "success",
            "collection": collection,
            "algorithm": algorithm,
            "num_clusters": num_clusters,
            "points_sampled": len(point_ids),
            "clusters": []
        }
        
        # Add cluster information
        for label, cluster in clusters.items():
            cluster_info = {
                "cluster_id": cluster["cluster_id"],
                "size": len(cluster["points"]),
                "points": cluster["points"][:10],  # Limit to 10 points per cluster
                "points_total": len(cluster["points"])
            }
            
            # Add topics if available
            if "topics" in cluster:
                cluster_info["topics"] = cluster["topics"]
            
            result["clusters"].append(cluster_info)
        
        # Sort clusters by size
        result["clusters"].sort(key=lambda x: x["size"], reverse=True)
        
        return result
    
    except Exception as e:
        await ctx.debug(f"Error performing clustering: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to perform clustering: {str(e)}"
        }

async def extract_cluster_topics(
    ctx: Context,
    texts: List[str],
    num_topics: int = 5
) -> List[Dict[str, Any]]:
    """
    Extract topics from a cluster of texts.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import nltk
        from nltk.corpus import stopwords
        
        # Download stopwords if needed
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        
        # Get stopwords
        stop_words = set(stopwords.words('english'))
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=stop_words
        )
        
        # Fit vectorizer
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Calculate the sum of TF-IDF scores for each term across all documents
        tfidf_sums = tfidf_matrix.sum(axis=0).A1
        
        # Get top terms
        top_indices = tfidf_sums.argsort()[-num_topics:][::-1]
        top_terms = [feature_names[i] for i in top_indices]
        
        # Return topics
        topics = []
        for i, term in enumerate(top_terms):
            topic = {
                "term": term,
                "weight": float(tfidf_sums[top_indices[i]])
            }
            topics.append(topic)
        
        return topics
    
    except Exception as e:
        await ctx.debug(f"Error extracting topics: {str(e)}")
        return []
```

### Integration with MCP Server

```python
@mcp.tool(name="semantic-clustering", description=tool_settings.semantic_clustering_description)
async def semantic_clustering_tool(ctx: Context, **kwargs) -> Dict[str, Any]:
    return await semantic_clustering(ctx, **kwargs)
```

### Dependencies

Add to `setup.py` or `pyproject.toml`:
```
scikit-learn>=1.0.2
hdbscan>=0.8.29
nltk>=3.7
```

## Real-time Data Connectors

### Overview

Create connectors to real-time data sources that can automatically update Qdrant collections.

### Implementation

```python
# tools/connectors/rss_connector.py
from typing import Dict, Any, List, Optional
import asyncio
import feedparser
import time
import hashlib
from datetime import datetime, timezone

from mcp.server.fastmcp import Context
from mcp_server_qdrant.tools.data_processing.chunk_and_process import chunk_and_process

async def setup_rss_connector(
    ctx: Context,
    feed_url: str,
    collection: str,
    check_interval: int = 3600,  # 1 hour
    max_items: int = 50,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    metadata: Optional[Dict[str, Any]] = None,
    run_immediately: bool = True,
) -> Dict[str, Any]:
    """
    Set up a connector to monitor an RSS feed and add new items to a collection.
    
    Parameters:
    -----------
    ctx : Context
        MCP request context
    feed_url : str
        URL of the RSS feed
    collection : str
        Collection to store items in
    check_interval : int, default=3600
        Interval between checks in seconds
    max_items : int, default=50
        Maximum number of items to process per check
    chunk_size : int, default=1000
        Size of chunks for processing
    chunk_overlap : int, default=200
        Overlap between chunks
    metadata : Dict[str, Any], optional
        Additional metadata to include with each item
    run_immediately : bool, default=True
        Whether to run the connector immediately
        
    Returns:
    --------
    Dict[str, Any]
        Connector configuration and status
    """
    await ctx.debug(f"Setting up RSS connector for feed: {feed_url}")
    
    # Initialize metadata if not provided
    if metadata is None:
        metadata = {}
    
    # Create connector ID
    connector_id = hashlib.md5(f"{feed_url}:{collection}".encode()).hexdigest()
    
    # Generate connector configuration
    connector_config = {
        "id": connector_id,
        "type": "rss",
        "feed_url": feed_url,
        "collection": collection,
        "check_interval": check_interval,
        "max_items": max_items,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "metadata": metadata,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "last_check": None,
        "last_item_date": None,
        "status": "configured"
    }
    
    try:
        # Store connector configuration
        # In a real implementation, this would be stored in a database
        # For demonstration, we'll just print it
        await ctx.debug(f"Connector configuration: {connector_config}")
        
        # Run connector immediately if requested
        if run_immediately:
            result = await process_rss_feed(
                ctx=ctx,
                feed_url=feed_url,
                collection=collection,
                max_items=max_items,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                metadata=metadata,
                last_item_date=None
            )
            
            connector_config["last_check"] = datetime.now(timezone.utc).isoformat()
            connector_config["last_item_date"] = result.get("last_item_date")
            connector_config["status"] = "active"
        
        return {
            "status": "success",
            "connector_id": connector_id,
            "connector_type": "rss",
            "feed_url": feed_url,
            "collection": collection,
            "check_interval": check_interval,
            "run_immediately": run_immediately,
            "initial_run_result": result if run_immediately else None
        }
    
    except Exception as e:
        await ctx.debug(f"Error setting up RSS connector: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to set up RSS connector: {str(e)}"
        }

async def process_rss_feed(
    ctx: Context,
    feed_url: str,
    collection: str,
    max_items: int = 50,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    metadata: Optional[Dict[str, Any]] = None,
    last_item_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process an RSS feed and add new items to a collection.
    """
    await ctx.debug(f"Processing RSS feed: {feed_url}")
    
    # Get Qdrant client
    qdrant_connector = ctx.request_context.lifespan_context["qdrant_connector"]
    
    # Initialize metadata if not provided
    if metadata is None:
        metadata = {}
    
    try:
        # Parse feed
        feed = feedparser.parse(feed_url)
        
        # Check feed title
        feed_title = feed.feed.get("title", "Unknown Feed")
        await ctx.debug(f"Feed title: {feed_title}")
        
        # Parse last item date
        last_date = None
        if last_item_date:
            try:
                last_date = datetime.fromisoformat(last_item_date.replace("Z", "+00:00"))
            except Exception:
                last_date = None
        
        # Process items
        processed_items = []
        last_processed_date = last_date
        
        for item in feed.entries[:max_items]:
            # Check if item is newer than last processed date
            item_date = None
            if "published_parsed" in item:
                item_date = datetime.fromtimestamp(time.mktime(item.published_parsed), timezone.utc)
            elif "updated_parsed" in item:
                item_date = datetime.fromtimestamp(time.mktime(item.updated_parsed), timezone.utc)
            
            # Skip if older than last processed date
            if last_date and item_date and item_date <= last_date:
                continue
            
            # Update last processed date
            if item_date and (not last_processed_date or item_date > last_processed_date):
                last_processed_date = item_date
            
            # Extract content
            title = item.get("title", "")
            description = item.get("description", "")
            content = item.get("content", [])
            content_text = ""
            if content:
                content_text = content[0].get("value", "")
            
            # Combine all text
            text = f"{title}\n\n{description}\n\n{content_text}"
            
            # Create item metadata
            item_metadata = {
                "source": "rss",
                "feed_url": feed_url,
                "feed_title": feed_title,
                "item_title": title,
                "item_link": item.get("link", ""),
                "item_date": item_date.isoformat() if item_date else None,
                "item_id": item.get("id", ""),
                **metadata
            }
            
            # Process item
            result = await chunk_and_process(
                ctx=ctx,
                text=text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                collection=collection,
                metadata=item_metadata
            )
            
            # Add to processed items
            processed_items.append({
                "title": title,
                "link": item.get("link", ""),
                "date": item_date.isoformat() if item_date else None,
                "chunks": result.get("count", 0)
            })
        
        return {
            "status": "success",
            "feed_url": feed_url,
            "feed_title": feed_title,
            "processed_items": len(processed_items),
            "items": processed_items,
            "last_item_date": last_processed_date.isoformat() if last_processed_date else None
        }
    
    except Exception as e:
        await ctx.debug(f"Error processing RSS feed: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to process RSS feed: {str(e)}"
        }
```

### Other Connector Tools

Additional real-time connectors could include:

1. **Twitter API Connector**
2. **Slack Channel Connector**
3. **Email Connector**
4. **Database Change Data Capture (CDC) Connector**

### Integration with MCP Server

```python
@mcp.tool(name="setup-rss-connector", description=tool_settings.setup_rss_connector_description)
async def setup_rss_connector_tool(ctx: Context, **kwargs) -> Dict[str, Any]:
    return await setup_rss_connector(ctx, **kwargs)
```

### Dependencies

Add to `setup.py` or `pyproject.toml`:
```
feedparser>=6.0.0
```

## Multilingual Support

### Overview

Enhance the Qdrant MCP server with multilingual capabilities for processing and searching content in different languages.

### Implementation

```python
# tools/multilingual/language_detection.py
from typing import Dict, Any, List, Optional
import re

from mcp.server.fastmcp import Context

async def detect_language(
    ctx: Context,
    text: str,
    confidence_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Detect the language of text.
    
    Parameters:
    -----------
    ctx : Context
        MCP request context
    text : str
        Text to detect language for
    confidence_threshold : float, default=0.5
        Minimum confidence threshold for detection
        
    Returns:
    --------
    Dict[str, Any]
        Detected language information
    """
    await ctx.debug(f"Detecting language for text ({len(text)} chars)")
    
    try:
        import langdetect
        from langdetect import DetectorFactory
        
        # Set seed for deterministic results
        DetectorFactory.seed = 0
        
        # Detect language
        result = langdetect.detect_langs(text)
        
        # Format results
        languages = []
        detected_language = None
        
        for lang in result:
            lang_code = lang.lang
            confidence = lang.prob
            
            languages.append({
                "language_code": lang_code,
                "confidence": confidence
            })
            
            # Set detected language if confidence exceeds threshold
            if confidence >= confidence_threshold and detected_language is None:
                detected_language = lang_code
        
        # Sort by confidence
        languages.sort(key=lambda x: x["confidence"], reverse=True)
        
        return {
            "status": "success",
            "detected_language": detected_language,
            "languages": languages,
            "text_length": len(text)
        }
    
    except Exception as e:
        await ctx.debug(f"Error detecting language: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to detect language: {str(e)}"
        }
```

### Translation Tool

```python
# tools/multilingual/translate.py
from typing import Dict, Any, List, Optional

from mcp.server.fastmcp import Context

async def translate_text(
    ctx: Context,
    text: str,
    target_language: str,
    source_language: Optional[str] = None,
    preserve_formatting: bool = True,
) -> Dict[str, Any]:
    """
    Translate text to a target language.
    
    Parameters:
    -----------
    ctx : Context
        MCP request context
    text : str
        Text to translate
    target_language : str
        Target language code (ISO 639-1)
    source_language : str, optional
        Source language code (if known)
    preserve_formatting : bool, default=True
        Whether to preserve formatting in translation
        
    Returns:
    --------
    Dict[str, Any]
        Translated text and metadata
    """
    await ctx.debug(f"Translating {len(text)} chars to {target_language}")
    
    try:
        from deep_translator import GoogleTranslator
        
        # Detect source language if not provided
        if not source_language:
            detected = await detect_language(ctx, text[:1000])  # Only use first 1000 chars for detection
            source_language = detected.get("detected_language", "auto")
        
        # Initialize translator
        translator = GoogleTranslator(source=source_language, target=target_language)
        
        # Handle long text by splitting into chunks
        if len(text) > 5000:
            chunks = split_text_for_translation(text, 5000, preserve_formatting)
            translated_chunks = []
            
            for chunk in chunks:
                translated_chunk = translator.translate(chunk)
                translated_chunks.append(translated_chunk)
            
            translated_text = "".join(translated_chunks)
        else:
            translated_text = translator.translate(text)
        
        return {
            "status": "success",
            "original_text": text,
            "translated_text": translated_text,
            "source_language": source_language,
            "target_language": target_language,
            "original_length": len(text),
            "translated_length": len(translated_text)
        }
    
    except Exception as e:
        await ctx.debug(f"Error translating text: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to translate text: {str(e)}"
        }

def split_text_for_translation(text: str, max_chunk_size: int, preserve_formatting: bool) -> List[str]:
    """Split text into chunks for translation while preserving structure."""
    if not preserve_formatting:
        # Simple splitting by character count
        return [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
    
    # Split text at paragraph boundaries to maintain formatting
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If adding this paragraph would exceed max size, start a new chunk
        if len(current_chunk) + len(paragraph) + 2 > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = paragraph
        else:
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
    
    # Add the last chunk if it has content
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks
