from typing import List
from app.services.hybrid_search import SearchResult
from app.core.config import settings
from app.core.logging import app_logger

_flashrank_model = None


def get_flashrank_model():
    global _flashrank_model
    if _flashrank_model is None:
        try:
            from flashrank import Ranker
            app_logger.info("Initializing local FlashRank Cross-Encoder reranker...")
            _flashrank_model = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="/tmp/flashrank")
        except Exception as e:
            app_logger.error(f"Failed to load FlashRank: {e}")
            raise
    return _flashrank_model


class RerankerService:
    """Reranks candidate search passages using Cross-Encoder models (FlashRank CPU default or Cohere API)."""

    @classmethod
    def rerank(cls, query: str, candidates: List[SearchResult], top_k: int = 5) -> List[SearchResult]:
        if not candidates:
            return []

        # Cohere Rerank API if configured
        if settings.RERANKER_PROVIDER == "cohere" and settings.COHERE_API_KEY:
            try:
                import httpx
                headers = {
                    "Authorization": f"Bearer {settings.COHERE_API_KEY}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": "rerank-english-v3.0",
                    "query": query,
                    "documents": [c.content for c in candidates],
                    "top_n": top_k,
                }
                with httpx.Client(timeout=10.0) as client:
                    resp = client.post("https://api.cohere.com/v1/rerank", json=payload, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()
                    reranked: List[SearchResult] = []
                    for item in data.get("results", []):
                        idx = item["index"]
                        res = candidates[idx]
                        res.score = float(item["relevance_score"])
                        reranked.append(res)
                    return reranked
            except Exception as e:
                app_logger.warning(f"Cohere rerank failed: {e}. Falling back to FlashRank.")

        # Default: Local FlashRank (Zero Cost, CPU-optimized)
        try:
            from flashrank import RerankRequest
            ranker = get_flashrank_model()
            passages = [{"id": str(i), "text": c.content} for i, c in enumerate(candidates)]
            rerank_request = RerankRequest(query=query, passages=passages)
            results = ranker.rerank(rerank_request)

            reranked_results: List[SearchResult] = []
            for item in results[:top_k]:
                orig_idx = int(item["id"])
                candidate = candidates[orig_idx]
                candidate.score = float(item["score"])
                reranked_results.append(candidate)
            return reranked_results
        except Exception as e:
            app_logger.error(f"Local FlashRank reranking error: {e}. Returning original candidates.")
            return candidates[:top_k]
