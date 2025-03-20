# Comprehensive Guide to Extending Qdrant MCP Server with Custom Tools

This guide provides a detailed roadmap for enhancing Qdrant MCP server with custom tools, making it a powerful vector database platform with advanced capabilities.

## Table of Contents

- [Introduction](#introduction)
- [Core Search and Query Tools](#core-search-and-query-tools)
- [Collection Management Tools](#collection-management-tools)
- [Data Processing Tools](#data-processing-tools)
- [Advanced Query Tools](#advanced-query-tools)
- [Analytics and Management Tools](#analytics-and-management-tools)
- [Document Processing Tools](#document-processing-tools)
- [Integration Tools](#integration-tools)
- [Implementation in MCP Config](#implementation-in-mcp-config)
- [Utility Functions](#utility-functions)
- [Setup and Deployment](#setup-and-deployment)

## Introduction

Qdrant is a powerful vector similarity search engine that can be significantly enhanced through custom tools in the MCP server. This guide outlines the implementation of various tools that extend Qdrant's capabilities for natural language processing, advanced queries, and data management.

## Core Search and Query Tools

### 1. Natural Language Query Tool

```typescript
// tools/nlq.ts
export async function naturalLanguageQuery(params: {
  query: string,
  collection: string,
  limit?: number,
  filter?: Record<string, any>,
  with_payload?: boolean | string[]
}) {
  const embedding = await generateEmbedding(params.query);
  const client = new QdrantClient({ host: 'localhost', port: 6333 });
  
  return await client.search(params.collection, {
    vector: embedding,
    limit: params.limit || 10,
    filter: params.filter,
    with_payload: params.with_payload || true
  });
}
```

### 2. Hybrid Search Tool

```typescript
export async function hybridSearch(params: {
  query: string,
  collection: string,
  limit?: number,
  filter?: Record<string, any>,
  textFieldName: string
}) {
  // Vector search
  const vectorResults = await naturalLanguageQuery({
    query: params.query,
    collection: params.collection,
    limit: params.limit ? params.limit * 3 : 30,
    filter: params.filter
  });
  
  // Keyword search via payload filter
  const keywordFilter = buildKeywordFilter(params.query, params.textFieldName);
  const keywordResults = await client.scroll(params.collection, {
    filter: keywordFilter,
    limit: params.limit ? params.limit * 3 : 30
  });
  
  // Combine and re-rank results
  return reRankResults(vectorResults, keywordResults, params.limit || 10);
}
```

### 3. Multi-Vector Search Tool

```typescript
export async function multiVectorSearch(params: {
  queries: string[],
  collection: string,
  weights?: number[],
  limit?: number
}) {
  const embeddings = await Promise.all(
    params.queries.map(query => generateEmbedding(query))
  );
  
  // Normalize weights if provided
  const weights = params.weights || embeddings.map(() => 1 / embeddings.length);
  
  return await client.search(params.collection, {
    vector: combineVectors(embeddings, weights),
    limit: params.limit || 10
  });
}
```

## Collection Management Tools

### 4. Collection Creation Tool

```typescript
export async function createCollection(params: {
  name: string,
  vector_size: number,
  distance?: 'Cosine' | 'Euclid' | 'Dot',
  hnsw_config?: Record<string, any>,
  optimizers_config?: Record<string, any>
}) {
  return await client.createCollection(params.name, {
    vectors: {
      size: params.vector_size,
      distance: params.distance || 'Cosine'
    },
    hnsw_config: params.hnsw_config,
    optimizers_config: params.optimizers_config
  });
}
```

### 5. Collection Migration Tool

```typescript
export async function migrateCollection(params: {
  source_collection: string,
  target_collection: string,
  batch_size?: number,
  transform_fn?: string
}) {
  const batchSize = params.batch_size || 100;
  let offset = null;
  const transformFn = params.transform_fn ? 
    eval(`(point) => { ${params.transform_fn} }`) : 
    (point) => point;
  
  // Paginated migration
  while (true) {
    const points = await client.scroll(params.source_collection, {
      limit: batchSize,
      offset
    });
    
    if (points.length === 0) break;
    
    const transformedPoints = points.map(transformFn);
    await client.upsert(params.target_collection, {
      points: transformedPoints
    });
    
    offset = points[points.length - 1].id;
  }
  
  return { status: "success", message: "Migration completed" };
}
```

## Data Processing Tools

### 6. Batch Embedding Tool

```typescript
export async function batchEmbed(params: {
  texts: string[],
  model?: string
}) {
  // Using an embedding API, e.g., OpenAI
  const embeddings = await Promise.all(
    params.texts.map(text => generateEmbedding(text, params.model))
  );
  
  return { embeddings };
}
```

### 7. Chunking and Processing Tool

```typescript
export async function chunkAndProcess(params: {
  text: string,
  chunk_size?: number,
  chunk_overlap?: number,
  collection?: string,
  metadata?: Record<string, any>
}) {
  const chunkSize = params.chunk_size || 1000;
  const overlap = params.chunk_overlap || 200;
  
  // Implement text chunking
  const chunks = textToChunks(params.text, chunkSize, overlap);
  
  const processed = await Promise.all(chunks.map(async (chunk, index) => {
    const embedding = await generateEmbedding(chunk);
    
    return {
      id: `chunk_${index}_${Date.now()}`,
      vector: embedding,
      payload: {
        text: chunk,
        chunk_index: index,
        total_chunks: chunks.length,
        ...params.metadata
      }
    };
  }));
  
  // If collection specified, upload directly
  if (params.collection) {
    await client.upsert(params.collection, {
      points: processed
    });
  }
  
  return { chunks: processed };
}
```

## Advanced Query Tools

### 8. Semantic Router Tool

```typescript
export async function semanticRouter(params: {
  query: string,
  routes: Array<{name: string, description: string}>,
  threshold?: number
}) {
  const queryEmbedding = await generateEmbedding(params.query);
  
  // Generate embeddings for each route description
  const routeEmbeddings = await Promise.all(
    params.routes.map(route => generateEmbedding(route.description))
  );
  
  // Calculate cosine similarity
  const similarities = routeEmbeddings.map(emb => 
    calculateCosineSimilarity(queryEmbedding, emb)
  );
  
  // Find best matching route
  const maxIndex = similarities.indexOf(Math.max(...similarities));
  const maxSimilarity = similarities[maxIndex];
  
  // Apply threshold if specified
  if (params.threshold && maxSimilarity < params.threshold) {
    return { route: null, confidence: maxSimilarity };
  }
  
  return { 
    route: params.routes[maxIndex].name, 
    confidence: maxSimilarity 
  };
}
```

### 9. Query Decomposition Tool

```typescript
export async function decomposeQuery(params: {
  query: string,
  use_llm?: boolean
}) {
  if (params.use_llm) {
    // Use LLM to decompose complex query into subqueries
    const decomposition = await callLLM(
      "Decompose this complex query into 2-4 simpler subqueries: " + params.query
    );
    return { subqueries: decomposition };
  } else {
    // Use rule-based decomposition
    return { subqueries: simpleDecompose(params.query) };
  }
}
```

## Analytics and Management Tools

### 10. Collection Analytics Tool

```typescript
export async function analyzeCollection(params: {
  collection: string,
  sample_size?: number
}) {
  // Get collection info
  const info = await client.getCollection(params.collection);
  
  // Sample points for analysis
  const sample = await client.scroll(params.collection, {
    limit: params.sample_size || 100
  });
  
  // Extract payload schema
  const payloadSchema = derivePayloadSchema(sample);
  
  // Calculate vector statistics
  const vectorStats = calculateVectorStats(sample);
  
  return {
    collection_info: info,
    count: info.vectors_count,
    payload_schema: payloadSchema,
    vector_stats: vectorStats
  };
}
```

### 11. Query Performance Tool

```typescript
export async function benchmarkQuery(params: {
  query: string,
  collection: string,
  iterations?: number
}) {
  const iterations = params.iterations || 5;
  const times = [];
  
  for (let i = 0; i < iterations; i++) {
    const start = performance.now();
    await naturalLanguageQuery({
      query: params.query,
      collection: params.collection
    });
    times.push(performance.now() - start);
  }
  
  return {
    avg_time_ms: times.reduce((a, b) => a + b, 0) / times.length,
    min_time_ms: Math.min(...times),
    max_time_ms: Math.max(...times),
    times_ms: times
  };
}
```

## Document Processing Tools

### 12. Document Indexing Tool

```typescript
export async function indexDocument(params: {
  document_url: string,
  collection: string,
  metadata?: Record<string, any>
}) {
  // Fetch document
  const document = await fetchDocument(params.document_url);
  
  // Extract text
  const text = await extractText(document);
  
  // Process chunks
  return await chunkAndProcess({
    text,
    collection: params.collection,
    metadata: {
      source_url: params.document_url,
      indexed_at: new Date().toISOString(),
      ...params.metadata
    }
  });
}
```

### 13. PDF Processing Tool

```typescript
export async function processPdf(params: {
  pdf_path: string,
  collection?: string,
  extract_tables?: boolean,
  extract_images?: boolean
}) {
  // Implementation depends on PDF processing library
  const textContent = await extractTextFromPdf(params.pdf_path);
  const metadata = await extractPdfMetadata(params.pdf_path);
  
  // Optional extractions
  let tables = [];
  let images = [];
  
  if (params.extract_tables) {
    tables = await extractTablesFromPdf(params.pdf_path);
  }
  
  if (params.extract_images) {
    images = await extractImagesFromPdf(params.pdf_path);
  }
  
  // Process for vector search if collection specified
  if (params.collection) {
    await chunkAndProcess({
      text: textContent,
      collection: params.collection,
      metadata: {
        source_type: "pdf",
        source_path: params.pdf_path,
        ...metadata
      }
    });
  }
  
  return {
    metadata,
    text_length: textContent.length,
    tables_count: tables.length,
    images_count: images.length
  };
}
```

## Integration Tools

### 14. Webhook Notification Tool

```typescript
export async function setupWebhook(params: {
  collection: string,
  event_types: Array<'create' | 'update' | 'delete'>,
  webhook_url: string,
  secret_key?: string
}) {
  // Register webhook in persistent storage
  const webhook = {
    id: generateId(),
    collection: params.collection,
    event_types: params.event_types,
    url: params.webhook_url,
    secret: params.secret_key,
    created_at: new Date().toISOString()
  };
  
  await storeWebhook(webhook);
  
  return { webhook_id: webhook.id };
}
```

### 15. Cross-Collection Join Tool

```typescript
export async function crossCollectionQuery(params: {
  primary_query: { query: string, collection: string },
  secondary_query: { collection: string, join_field: string },
  limit?: number
}) {
  // Query primary collection
  const primaryResults = await naturalLanguageQuery({
    query: params.primary_query.query,
    collection: params.primary_query.collection,
    limit: params.limit
  });
  
  // Extract join values
  const joinValues = primaryResults.map(
    result => result.payload[params.secondary_query.join_field]
  );
  
  // Query secondary collection with filter
  const secondaryResults = await client.scroll(
    params.secondary_query.collection,
    {
      filter: {
        must: [
          {
            key: "id",
            match: { any: joinValues }
          }
        ]
      }
    }
  );
  
  // Join results
  return joinResults(primaryResults, secondaryResults, 
                     params.secondary_query.join_field);
}
```

## Implementation in MCP Config

```javascript
// mcp-config.js
module.exports = {
  "tools": [
    {
      "name": "nlq_search",
      "description": "Search Qdrant using natural language",
      "function": "./tools/nlq.ts:naturalLanguageQuery"
    },
    {
      "name": "hybrid_search",
      "description": "Perform hybrid vector and keyword search",
      "function": "./tools/hybrid_search.ts:hybridSearch"
    },
    {
      "name": "multi_vector_search",
      "description": "Search using multiple query vectors with weights",
      "function": "./tools/multi_vector.ts:multiVectorSearch"
    },
    {
      "name": "create_collection",
      "description": "Create a new vector collection with specified parameters",
      "function": "./tools/collection_mgmt.ts:createCollection"
    },
    {
      "name": "migrate_collection",
      "description": "Migrate data between collections with optional transformations",
      "function": "./tools/collection_mgmt.ts:migrateCollection"
    },
    // Continue with all other tools...
  ]
}
```

## Utility Functions

These shared functions would be placed in a utilities file:

```typescript
// tools/utils.ts

// Vector operations
export async function generateEmbedding(text, model = "default") {
  // Implementation using your preferred embedding model
}

export function combineVectors(vectors, weights) {
  // Weighted vector combination
}

export function calculateCosineSimilarity(vec1, vec2) {
  // Cosine similarity calculation
}

// Text processing
export function textToChunks(text, chunkSize, overlap) {
  // Text chunking implementation
}

export function buildKeywordFilter(query, fieldName) {
  // Convert query to keyword filter
}

// Results processing
export function reRankResults(vectorResults, keywordResults, limit) {
  // Combine and re-rank results
}

export function joinResults(primaryResults, secondaryResults, joinField) {
  // Join results from two collections
}

// Document processing
export async function fetchDocument(url) {
  // Fetch document from URL
}

export async function extractText(document) {
  // Extract text from document
}

// Analytics
export function derivePayloadSchema(samples) {
  // Derive schema from samples
}

export function calculateVectorStats(samples) {
  // Calculate vector statistics
}
```

## Setup and Deployment

### Directory Structure
```
/qdrant-mcp-extended
  /tools
    nlq.ts
    hybrid_search.ts
    multi_vector.ts
    collection_mgmt.ts
    data_processing.ts
    advanced_query.ts
    analytics.ts
    document_processing.ts
    integration.ts
    utils.ts
  mcp-config.js
  package.json
  Dockerfile
```

### Dependencies
```json
{
  "dependencies": {
    "@qdrant/js-client-rest": "^1.0.0",
    "openai": "^4.0.0",
    "node-fetch": "^3.3.0",
    "pdf-parse": "^1.1.1"
  }
}
```

### Docker Setup
```dockerfile
FROM node:18
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

### Docker Compose Integration
```yaml
version: '3'
services:
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - ./qdrant_data:/qdrant/storage
  
  qdrant-mcp:
    build: ./qdrant-mcp-extended
    ports:
      - "3000:3000"
    environment:
      - QDRANT_URL=http://qdrant:6333
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - qdrant
```

This comprehensive set of tools transforms your Qdrant MCP server into a powerful platform capable of handling a wide range of vector database operations with sophisticated natural language querying capabilities.
