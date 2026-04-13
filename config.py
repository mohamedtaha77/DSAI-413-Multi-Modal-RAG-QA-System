from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ColPali embedding model
    colpali_model: str = "vidore/colqwen2.5-v0.2"

    # Qdrant — leave QDRANT_URL empty to use local persistent storage
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    collection_name: str = "rag_pages"

    # Kimi K2 via NVIDIA NIM — get a key at build.nvidia.com
    kimi_api_key: str = ""
    kimi_model: str = "meta/llama-4-maverick-17b-128e-instruct"
    kimi_base_url: str = "https://integrate.api.nvidia.com/v1"

    # Retrieval
    top_k: int = 3

    # PDF rendering resolution
    dpi: int = 150

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
