import os
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ENVIRONMENT: str = "production"
    PROJECT_NAME: str = "Enterprise Document Intelligence & RAG API"
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = "dev-secret-key-change-in-production"

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/rag_db"
    SYNC_DATABASE_URL: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/rag_db"

    # Redis & Celery
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"

    # Embeddings
    EMBEDDING_PROVIDER: Literal["fastembed", "openai"] = "fastembed"
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIMENSION: int = 384

    # LLM
    LLM_PROVIDER: Literal["openai", "gemini"] = "openai"
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-1.5-flash"

    # Reranker
    RERANKER_PROVIDER: Literal["flashrank", "cohere"] = "flashrank"
    COHERE_API_KEY: str = ""

    # Ingestion Parameters
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100
    MAX_UPLOAD_SIZE_MB: int = 25

    # Multi-tenancy Admin
    MASTER_API_KEY: str = "enterprise-super-secret-admin-key"

    @property
    def async_database_url(self) -> str:
        """Sanitizes Render / Heroku / Railway connection strings for SQLAlchemy asyncpg."""
        url = self.DATABASE_URL
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+asyncpg://", 1)
        elif url.startswith("postgresql://") and "+asyncpg" not in url:
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return url

    @property
    def sync_database_url(self) -> str:
        """Sanitizes connection strings for synchronous Celery worker / psycopg2."""
        url = self.SYNC_DATABASE_URL or self.DATABASE_URL
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+psycopg2://", 1)
        elif url.startswith("postgresql://") and "+psycopg2" not in url:
            url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
        return url

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
