"""Application configuration management using environment variables."""

from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """ShopEase application configuration settings."""

    # Application details
    app_name: str = "ShopEase AI Support"
    app_env: str = "development"
    debug: bool = True
    port: int = 8000
    host: str = "127.0.0.1"

    # SQLite Database connection string
    database_url: str = "sqlite:///./shopease.db"

    # Vector Database & RAG Configuration
    chroma_persist_dir: str = "./chroma_db"
    kb_dir: str = "./knowledge_base"
    embedding_model: str = "all-MiniLM-L6-v2"
    retrieval_threshold: float = 0.60  # Max cosine distance used to derive the semantic relevance floor
    top_k_retrieval: int = 3

    # Gemini LLM configuration (Optional until LLM stage)
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-3.8-flash"
    gemini_timeout_seconds: float = 20.0

    # Logging
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings instance."""
    return Settings()
