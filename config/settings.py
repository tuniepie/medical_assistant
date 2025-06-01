from functools import lru_cache
from pydantic_settings  import BaseSettings, SettingsConfigDict  

class Settings(BaseSettings):
    """Application settings"""
    
    cohere_api_key: str
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_k: int = 3
    max_tokens: int = 1000
    temperature: float = 0.1
    llm: str = "command-r-plus"
    embedding_model_name: str = "embed-english-v3.0"
    enable_file_logging: bool = False
    
    model_config = SettingsConfigDict(env_file=".env")

@lru_cache()
def get_settings():
    return Settings()