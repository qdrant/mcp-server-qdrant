from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from openai import AsyncOpenAI
import asyncio


class OAICompatProvider(EmbeddingProvider):
    def __init__(self, model_name: str, endpoint_url: str, api_key: str, vec_size: int | None = None) -> None:
        self.model_name = model_name
        self.base_url = endpoint_url
        self.key = api_key
        self.oai_client = AsyncOpenAI(base_url=self.base_url, api_key=self.key)
        self._vec_size = self.fetch_vec_size() if vec_size is None else vec_size
    
    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        response = await self.oai_client.embeddings.create(
            input=documents,
            model=self.model_name
        )
        results = [embed.embedding for embed in response.data]
        return results
    
    async def embed_query(self, query: str) -> list[float]:
        response = await self.oai_client.embeddings.create(
            input=query,
            model=self.model_name
        )
        results = response.data[0].embedding
        return results
    
    def get_vector_name(self) -> str:
        return f"oai_compat-{self.model_name}"

    def get_vector_size(self) -> int:
        return self._vec_size
    
    def fetch_vec_size(self):
        """Retrieves the embedding vector dimensionality by sending a minimal inference request to the endpoint.
        Since the OpenAI-compatible API does not expose model metadata detailing vector dimensions, this method
        uses a dummy input to obtain a sample embedding and derives the dimension count from its structure.
        To avoid unnecessary latency and cost during initialization, it is recommended to explicitly provide
        the vector size via the `vec_size` parameter when the dimension is known."""
        try:
            loop = asyncio.get_running_loop()
            fut = asyncio.run_coroutine_threadsafe(self.embed_query("dummy"), loop)
            result = fut.result()
        except RuntimeError:
            result = asyncio.run(self.embed_query("dummy"))
        return len(result)