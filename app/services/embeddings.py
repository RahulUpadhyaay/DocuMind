from typing import List
from app.core.config import settings
from app.core.logging import app_logger

# Lazy singleton instances to avoid reload overhead
_fastembed_model = None
_openai_client = None


def get_fastembed_model():
    global _fastembed_model
    if _fastembed_model is None:
        try:
            from fastembed import TextEmbedding
            app_logger.info(f"Loading local FastEmbed model: {settings.EMBEDDING_MODEL} (Zero cost, CPU-optimized)")
            _fastembed_model = TextEmbedding(model_name=settings.EMBEDDING_MODEL)
        except Exception as e:
            app_logger.error(f"Failed to initialize FastEmbed: {e}")
            raise
    return _fastembed_model


def get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _openai_client


class EmbeddingService:
    """Enterprise Embedding Service supporting local FastEmbed (default zero-cost) and OpenAI."""

    @classmethod
    def generate_embeddings(cls, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        if settings.EMBEDDING_PROVIDER == "openai" and settings.OPENAI_API_KEY:
            try:
                client = get_openai_client()
                response = client.embeddings.create(
                    input=texts,
                    model=settings.EMBEDDING_MODEL if "text-embedding" in settings.EMBEDDING_MODEL else "text-embedding-3-small",
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                app_logger.warning(f"OpenAI embedding failed ({e}), falling back to local FastEmbed.")

        # Default: FastEmbed (Local, CPU, Fast, 0-cost)
        model = get_fastembed_model()
        embeddings_generator = model.embed(texts)
        return [embedding.tolist() for embedding in embeddings_generator]

    @classmethod
    def generate_query_embedding(cls, query: str) -> List[float]:
        return cls.generate_embeddings([query])[0]
