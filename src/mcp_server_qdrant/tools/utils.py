"""
Utility functions for Qdrant MCP server tools.
"""
import asyncio
import json
import re
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import httpx
from qdrant_client.http.models import Filter

from mcp_server_qdrant.qdrant import QdrantConnector


async def combine_vectors(vectors: List[List[float]], weights: Optional[List[float]] = None) -> List[float]:
    """
    Combine multiple vectors with optional weights.
    
    Parameters:
    -----------
    vectors : List[List[float]]
        List of vectors to combine
    weights : List[float], optional
        Weights for each vector (defaults to equal weights)
        
    Returns:
    --------
    List[float]
        Combined vector
    """
    if not vectors:
        raise ValueError("No vectors provided")
    
    # Default to equal weights if not provided
    if weights is None:
        weights = [1.0 / len(vectors)] * len(vectors)
    
    if len(vectors) != len(weights):
        raise ValueError("Number of vectors and weights must match")
    
    # Combine vectors with weights
    combined = np.zeros_like(vectors[0], dtype=np.float32)
    for vec, weight in zip(vectors, weights):
        combined += np.array(vec) * weight
    
    # Normalize the combined vector
    norm = np.linalg.norm(combined)
    if norm > 0:
        combined = combined / norm
    
    return combined.tolist()


def calculate_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Parameters:
    -----------
    vec1 : List[float]
        First vector
    vec2 : List[float]
        Second vector
        
    Returns:
    --------
    float
        Cosine similarity (ranges from -1 to 1)
    """
    if not vec1 or not vec2:
        raise ValueError("Vectors must not be empty")
    
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same dimensionality")
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm_a = np.sqrt(sum(a * a for a in vec1))
    norm_b = np.sqrt(sum(b * b for b in vec2))
    
    if norm_a == 0 or norm_b == 0:
        return 0
    
    return dot_product / (norm_a * norm_b)


def text_to_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Split text into chunks with specified size and overlap.
    
    Parameters:
    -----------
    text : str
        Text to split into chunks
    chunk_size : int, default=1000
        Maximum size of each chunk
    chunk_overlap : int, default=200
        Overlap between consecutive chunks
        
    Returns:
    --------
    List[str]
        List of text chunks
    """
    if not text:
        return []
    
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive")
    
    if chunk_overlap < 0:
        raise ValueError("Chunk overlap must be non-negative")
    
    if chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be less than chunk size")
    
    # Simple splitting by character count
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        
        # Try to find a natural breakpoint (sentence or paragraph end)
        if end < len(text):
            # Look for paragraph break
            paragraph_end = text.rfind('\n\n', start, end)
            if paragraph_end != -1 and paragraph_end > start + chunk_size // 2:
                end = paragraph_end + 2
            else:
                # Look for sentence end
                sentence_end = max(
                    text.rfind('. ', start, end),
                    text.rfind('! ', start, end),
                    text.rfind('? ', start, end)
                )
                if sentence_end != -1 and sentence_end > start + chunk_size // 2:
                    end = sentence_end + 2
        
        chunks.append(text[start:end])
        start = end - chunk_overlap
    
    return chunks


def build_keyword_filter(query: str, field_name: str) -> Dict[str, Any]:
    """
    Build a Qdrant filter for keyword search in a text field.
    
    Parameters:
    -----------
    query : str
        Search query
    field_name : str
        Name of the field to search in
        
    Returns:
    --------
    Dict[str, Any]
        Qdrant filter structure
    """
    # Extract important keywords from the query
    # This is a simple approach - remove stop words and split by space
    stop_words = {"a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by"}
    words = query.lower().split()
    keywords = [word for word in words if word not in stop_words and len(word) > 2]
    
    # If no valid keywords, return a simple match all filter
    if not keywords:
        return {}
    
    # Create a filter that matches any of the keywords
    return {
        "should": [
            {
                "key": field_name,
                "match": {"text": keyword}
            }
            for keyword in keywords
        ]
    }


async def re_rank_results(vector_results: List[Dict[str, Any]], 
                         keyword_results: List[Dict[str, Any]], 
                         limit: int = 10) -> List[Dict[str, Any]]:
    """
    Combine and re-rank results from vector and keyword searches.
    
    Parameters:
    -----------
    vector_results : List[Dict[str, Any]]
        Results from vector search
    keyword_results : List[Dict[str, Any]]
        Results from keyword search
    limit : int, default=10
        Maximum number of results to return
        
    Returns:
    --------
    List[Dict[str, Any]]
        Combined and re-ranked results
    """
    # Create a dictionary to combine scores
    combined_results = {}
    
    # Add vector search results (with weight 0.7)
    for result in vector_results:
        result_id = str(result["id"])
        combined_results[result_id] = {
            **result,
            "combined_score": result["score"] * 0.7
        }
    
    # Add keyword search results (with weight 0.3)
    for result in keyword_results:
        result_id = str(result["id"])
        if result_id in combined_results:
            # If already exists from vector search, add the scores
            combined_results[result_id]["combined_score"] += result.get("score", 0.5) * 0.3
        else:
            # Otherwise, add with keyword score only
            combined_results[result_id] = {
                **result,
                "combined_score": result.get("score", 0.5) * 0.3
            }
    
    # Convert to list and sort by combined score
    result_list = list(combined_results.values())
    result_list.sort(key=lambda x: x["combined_score"], reverse=True)
    
    # Return only the requested number of results
    return result_list[:limit]


async def join_results(primary_results: List[Dict[str, Any]], 
                     secondary_results: List[Dict[str, Any]], 
                     join_field: str) -> List[Dict[str, Any]]:
    """
    Join results from primary and secondary collections.
    
    Parameters:
    -----------
    primary_results : List[Dict[str, Any]]
        Results from primary collection
    secondary_results : List[Dict[str, Any]]
        Results from secondary collection
    join_field : str
        Field name to join on
        
    Returns:
    --------
    List[Dict[str, Any]]
        Joined results
    """
    # Create a lookup map for secondary results
    secondary_map = {str(result["id"]): result for result in secondary_results}
    
    # Join with primary results
    joined_results = []
    for primary in primary_results:
        join_value = primary.get("payload", {}).get(join_field)
        
        if join_value and str(join_value) in secondary_map:
            # Add secondary data to primary result
            joined_results.append({
                **primary,
                "joined_data": secondary_map[str(join_value)]
            })
        else:
            # Include primary data without join
            joined_results.append({
                **primary,
                "joined_data": None
            })
    
    return joined_results


async def fetch_document(url: str) -> str:
    """
    Fetch document content from URL.
    
    Parameters:
    -----------
    url : str
        URL to fetch
        
    Returns:
    --------
    str
        Document content
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.text


async def extract_text(document: str) -> str:
    """
    Extract plain text from document.
    This is a simplified version that assumes document is already text.
    For HTML or other formats, additional processing would be needed.
    
    Parameters:
    -----------
    document : str
        Document content
        
    Returns:
    --------
    str
        Extracted text
    """
    # Check if document is HTML
    if document.strip().startswith("<!DOCTYPE html>") or document.strip().startswith("<html"):
        # Very basic HTML tag removal - a real implementation would use a proper HTML parser
        text = re.sub(r'<[^>]+>', ' ', document)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    # Return as is for plain text
    return document


def derive_payload_schema(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract payload schema from sample points.
    
    Parameters:
    -----------
    samples : List[Dict[str, Any]]
        Sample points from collection
        
    Returns:
    --------
    Dict[str, Any]
        Schema of payload fields and their types
    """
    if not samples:
        return {}
    
    schema = {}
    
    for sample in samples:
        payload = sample.get("payload", {})
        for key, value in payload.items():
            if key not in schema:
                schema[key] = {"type": type(value).__name__}
                
                # Add example value
                if isinstance(value, (int, float, bool, str)):
                    schema[key]["example"] = value
                elif isinstance(value, (list, dict)):
                    schema[key]["example"] = "complex structure"
    
    return schema


def calculate_vector_stats(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate vector statistics from sample points.
    
    Parameters:
    -----------
    samples : List[Dict[str, Any]]
        Sample points from collection
        
    Returns:
    --------
    Dict[str, Any]
        Vector statistics
    """
    if not samples:
        return {}
    
    # Extract vectors from samples
    vectors = []
    for sample in samples:
        if "vector" in sample:
            vector = sample["vector"]
            if isinstance(vector, dict):
                # Handle named vectors
                for vec_name, vec_values in vector.items():
                    if isinstance(vec_values, list) and all(isinstance(v, (int, float)) for v in vec_values):
                        vectors.append((vec_name, vec_values))
            elif isinstance(vector, list) and all(isinstance(v, (int, float)) for v in vector):
                # Handle unnamed vector
                vectors.append(("default", vector))
    
    if not vectors:
        return {"error": "No valid vectors found in samples"}
    
    # Group vectors by name
    vectors_by_name = {}
    for name, vec in vectors:
        if name not in vectors_by_name:
            vectors_by_name[name] = []
        vectors_by_name[name].append(vec)
    
    # Calculate statistics for each vector group
    stats = {}
    for name, vecs in vectors_by_name.items():
        # Convert to numpy for easier calculations
        np_vecs = np.array(vecs)
        
        stats[name] = {
            "dimensions": np_vecs.shape[1],
            "count": len(vecs),
            "min": np.min(np_vecs).item(),
            "max": np.max(np_vecs).item(),
            "mean": np.mean(np_vecs).item(),
            "std": np.std(np_vecs).item(),
            "l2_norm_mean": np.mean([np.linalg.norm(v) for v in np_vecs]).item()
        }
    
    return stats


async def simple_decompose(query: str) -> List[str]:
    """
    Simple rule-based query decomposition.
    
    Parameters:
    -----------
    query : str
        Complex query to decompose
        
    Returns:
    --------
    List[str]
        List of simpler subqueries
    """
    # Check for multiple questions
    if '?' in query:
        subqueries = [q.strip() + '?' for q in query.split('?') if q.strip()]
        if len(subqueries) > 1:
            return subqueries
    
    # Check for "and" splits
    if ' and ' in query.lower():
        and_splits = query.lower().split(' and ')
        if len(and_splits) > 1:
            return [query.split(' and ')[0]] + [s.strip() for s in and_splits[1:]]
    
    # Break down based on commas
    if ',' in query:
        comma_parts = query.split(',')
        if len(comma_parts) > 1:
            return [p.strip() for p in comma_parts]
    
    # Fallback: just return the original query
    return [query]


async def llm_decompose(query: str, llm_api_key: str) -> List[str]:
    """
    Decompose a complex query using an LLM.
    This is a placeholder - the real implementation would use your preferred LLM service.
    
    Parameters:
    -----------
    query : str
        Complex query to decompose
    llm_api_key : str
        API key for LLM service
        
    Returns:
    --------
    List[str]
        List of simpler subqueries
    """
    # Placeholder implementation
    # In a real implementation, you would call an LLM API here
    return await simple_decompose(query)  # Fallback to simple decomposition
