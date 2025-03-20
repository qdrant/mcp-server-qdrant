# Web Crawling Integration for Qdrant MCP

This document outlines a comprehensive approach to integrating web crawling capabilities directly into a Qdrant MCP server, providing a powerful way to index web content into vector collections.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Implementation Guide](#implementation-guide)
  - [Directory Structure](#directory-structure)
  - [Core Components](#core-components)
  - [Tool Implementations](#tool-implementations)
  - [Server Integration](#server-integration)
- [Usage Examples](#usage-examples)
- [Advanced Features](#advanced-features)
- [Dependencies](#dependencies)
- [Advantages Over External Solutions](#advantages-over-external-solutions)

## Architecture Overview

The web crawling integration for Qdrant MCP consists of five modular components that work together to provide flexible web content indexing:

1. **Basic URL Crawler** - Process single web pages and extract content
2. **Batch Crawler** - Handle multiple URLs with configurable parameters
3. **Recursive Crawler** - Deep crawling with customizable constraints
4. **Sitemap Parser** - Extract URLs from sitemap.xml for efficient crawling
5. **Content Processor** - Handle different content types (HTML, PDF, etc.)

All components leverage the existing chunking and embedding tools to ensure consistent processing and storage in the Qdrant vector database.

## Implementation Guide

### Directory Structure

```
/src/mcp_server_qdrant/tools/web/
  ├── __init__.py
  ├── crawl_url.py         # Single URL processing
  ├── batch_crawl.py       # Multiple URLs processing 
  ├── recursive_crawl.py   # Deep crawling with constraints
  ├── sitemap_extract.py   # Sitemap.xml parsing
  └── content_processor.py # Content type handling
```

### Core Components

#### Single URL Crawler

The foundation of the web crawling system is the `crawl_url` tool, which handles single page extraction:

- Fetches content from a specified URL
- Detects content type (HTML, PDF, etc.)
- Processes content based on type
- Extracts metadata (title, links, etc.)
- Chunks content appropriately
- Generates embeddings
- Stores in Qdrant collection

#### Recursive Crawler

The `recursive_crawl` tool builds on the single URL crawler to provide depth-first web crawling:

- Starts from a seed URL
- Follows links according to constraints
- Maintains a frontier of URLs to visit
- Tracks visited URLs to avoid duplicates
- Handles depth limits and domain restrictions
- Provides comprehensive crawl statistics

#### Sitemap Parser

The `sitemap_extract` tool provides an efficient way to crawl structured websites:

- Parses standard sitemap.xml files
- Handles sitemap index files
- Extracts URLs and metadata (priority, lastmod, etc.)
- Filters URLs based on priority
- Supports crawling extracted URLs
- Works with the single URL crawler

### Tool Implementations

#### URL Crawler Implementation

```python
async def crawl_url(
    ctx: Context,
    url: str,
    collection: str,
    extract_links: bool = False,
    follow_links: bool = False,
    max_links: int = 5,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Crawl a single URL, extract content, and store in Qdrant.
    """
    # 1. Fetch content with httpx
    # 2. Process based on content type (HTML, PDF, etc.)
    # 3. Extract metadata and links if requested
    # 4. Chunk and process the content
    # 5. Store in Qdrant collection
    # 6. Optionally follow links
```

#### Recursive Crawler Implementation

```python
async def recursive_crawl(
    ctx: Context,
    start_url: str,
    collection: str,
    max_pages: int = 20,
    max_depth: int = 3,
    stay_on_domain: bool = True,
    stay_on_path: bool = False,
    exclude_patterns: List[str] = None,
    include_patterns: List[str] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Recursively crawl a website and index content in Qdrant.
    """
    # 1. Parse start URL for domain/path constraints
    # 2. Initialize tracking (visited URLs, pending URLs)
    # 3. Iteratively crawl URLs up to max_pages
    # 4. Apply domain, path, and pattern filters
    # 5. Track and return comprehensive statistics
```

#### Sitemap Parser Implementation

```python
async def sitemap_extract(
    ctx: Context,
    sitemap_url: str,
    collection: str,
    max_urls: int = 50,
    priority_threshold: float = 0.0,
    crawl_extracted_urls: bool = True,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Extract URLs from a sitemap.xml file and optionally crawl them.
    """
    # 1. Fetch and parse sitemap XML
    # 2. Handle sitemap index (with sub-sitemaps)
    # 3. Extract URLs and metadata (priority, lastmod)
    # 4. Filter by priority threshold
    # 5. Optionally crawl extracted URLs
    # 6. Return comprehensive statistics
```

### Server Integration

To integrate these tools into the MCP server, add the following to your `server.py` file:

```python
# Add import statements
from mcp_server_qdrant.tools.web.crawl_url import crawl_url
from mcp_server_qdrant.tools.web.recursive_crawl import recursive_crawl
from mcp_server_qdrant.tools.web.sitemap_extract import sitemap_extract
from mcp_server_qdrant.tools.web.batch_crawl import batch_crawl

# Register web crawling tools
@mcp.tool(name="crawl-url", description=tool_settings.crawl_url_description)
async def crawl_url_tool(ctx: Context, **kwargs) -> Dict[str, Any]:
    return await crawl_url(ctx, **kwargs)

@mcp.tool(name="recursive-crawl", description=tool_settings.recursive_crawl_description)
async def recursive_crawl_tool(ctx: Context, **kwargs) -> Dict[str, Any]:
    return await recursive_crawl(ctx, **kwargs)

@mcp.tool(name="sitemap-extract", description=tool_settings.sitemap_extract_description)
async def sitemap_extract_tool(ctx: Context, **kwargs) -> Dict[str, Any]:
    return await sitemap_extract(ctx, **kwargs)

@mcp.tool(name="batch-crawl", description=tool_settings.batch_crawl_description)
async def batch_crawl_tool(ctx: Context, **kwargs) -> Dict[str, Any]:
    return await batch_crawl(ctx, **kwargs)
```

Update your `settings.py` to include descriptions for these tools:

```python
# Add to DEFAULT_TOOL_DESCRIPTIONS
DEFAULT_TOOL_DESCRIPTIONS = {
    # Existing tools...
    
    # Web crawling tools
    "crawl_url": "Crawl a single URL and store content in Qdrant",
    "recursive_crawl": "Recursively crawl a website and store content in Qdrant",
    "sitemap_extract": "Extract and process URLs from a sitemap.xml file",
    "batch_crawl": "Process multiple URLs in batch and store in Qdrant",
}

# Add to ToolSettings class
crawl_url_description: str = Field(
    default=DEFAULT_TOOL_DESCRIPTIONS["crawl_url"],
    validation_alias="CRAWL_URL_DESCRIPTION",
)
recursive_crawl_description: str = Field(
    default=DEFAULT_TOOL_DESCRIPTIONS["recursive_crawl"],
    validation_alias="RECURSIVE_CRAWL_DESCRIPTION",
)
sitemap_extract_description: str = Field(
    default=DEFAULT_TOOL_DESCRIPTIONS["sitemap_extract"],
    validation_alias="SITEMAP_EXTRACT_DESCRIPTION",
)
batch_crawl_description: str = Field(
    default=DEFAULT_TOOL_DESCRIPTIONS["batch_crawl"],
    validation_alias="BATCH_CRAWL_DESCRIPTION",
)
```

## Usage Examples

### Basic URL Crawling

```python
# Crawl a single page
result = await crawl_url(
    ctx=ctx,
    url="https://example.com/article",
    collection="web_content",
    extract_links=True,
    metadata={"source": "manual", "category": "example"}
)
```

### Recursive Crawling

```python
# Crawl a website recursively
result = await recursive_crawl(
    ctx=ctx,
    start_url="https://example.com/blog/",
    collection="blog_content",
    max_pages=50,
    max_depth=3,
    stay_on_domain=True,
    stay_on_path=True,
    exclude_patterns=["author", "tag", "category", "archive"],
    metadata={"source": "blog", "crawl_id": "weekly-update"}
)
```

### Sitemap Processing

```python
# Extract and process URLs from sitemap
result = await sitemap_extract(
    ctx=ctx,
    sitemap_url="https://example.com/sitemap.xml",
    collection="website_content",
    max_urls=100,
    priority_threshold=0.5,
    crawl_extracted_urls=True,
    metadata={"source": "sitemap", "sitemap_date": "2023-03-19"}
)
```

### Batch URL Processing

```python
# Process multiple URLs in batch
result = await batch_crawl(
    ctx=ctx,
    urls=[
        "https://example.com/page1",
        "https://example.com/page2",
        "https://anotherexample.com/article"
    ],
    collection="mixed_content",
    parallel_requests=3,
    chunk_size=1500,
    metadata={"batch_id": "manual-selection-2023-03"}
)
```

## Advanced Features

### Rate Limiting and Politeness

```python
# Configure crawling with rate limiting
result = await recursive_crawl(
    ctx=ctx,
    start_url="https://example.com",
    collection="polite_crawl",
    max_pages=30,
    # Rate limiting settings
    respect_robots_txt=True,
    request_delay=2.0,  # seconds between requests
    max_requests_per_domain=10,
    retry_count=3,
    retry_delay=5.0,
    # Other settings...
)
```

### Content Type Handling

The content processor handles different file types:

- **HTML**: Extract main content, remove navigation, extract metadata
- **PDF**: Extract text, TOC, tables, and optionally images
- **Office Documents**: Extract text from Word, Excel, PowerPoint (with appropriate libraries)
- **Plain Text**: Process directly with minimal transformation
- **JSON/XML**: Extract structured data

### Authentication Support

```python
# Crawl authenticated content
result = await crawl_url(
    ctx=ctx,
    url="https://example.com/protected-page",
    collection="authenticated_content",
    # Authentication settings
    auth={
        "type": "basic",  # or "digest", "bearer", "custom"
        "username": "user",
        "password": "pass"
    },
    # Or
    headers={
        "Authorization": "Bearer token123",
        "Cookie": "session=abc123"
    },
    # Other settings...
)
```

### Incremental Crawling

```python
# Perform incremental updates
result = await sitemap_extract(
    ctx=ctx,
    sitemap_url="https://example.com/sitemap.xml",
    collection="incremental_content",
    # Incremental settings
    incremental=True,
    update_strategy="modified_only",  # or "all", "new_only"
    check_lastmod=True,
    # Other settings...
)
```

## Dependencies

Required Python packages:

```
httpx>=0.23.0
beautifulsoup4>=4.10.0
lxml>=4.9.0
PyPDF2>=3.0.0
python-dateutil>=2.8.2
```

Optional dependencies:

```
# For office documents
python-docx>=0.8.11
openpyxl>=3.0.10
python-pptx>=0.6.21

# For advanced PDF processing
pdfplumber>=0.7.0
tabula-py>=2.3.0
```

## Advantages Over External Solutions

1. **Direct Integration**
   - Seamless part of your MCP server
   - No need to transfer data between systems
   - Consistent interface and error handling

2. **Optimized for Qdrant**
   - Built specifically for vector search
   - Designed for semantic similarity matching
   - Leverages existing embedding and chunking tools

3. **Customization**
   - Full control over crawling parameters
   - Customizable processing pipeline
   - Flexible metadata handling

4. **Efficiency**
   - No duplicate processing or storage
   - Shared resources with existing MCP server
   - Optimized chunking for vector search

5. **Unified Experience**
   - Same interface for all data ingestion
   - Consistent metadata and filtering
   - Simplified infrastructure management
