from typing import List, Dict, Any, Optional
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc, text
from pgvector.sqlalchemy import Vector

from app.db.models import DocumentChunk, Document
from app.services.embeddings import EmbeddingService
from app.core.logging import app_logger


class SearchResult:
    def __init__(
        self,
        chunk_id: uuid.UUID,
        document_id: uuid.UUID,
        filename: str,
        page_number: int,
        content: str,
        score: float,
        source: str = "hybrid",
    ):
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.filename = filename
        self.page_number = page_number
        self.content = content
        self.score = score
        self.source = source


class HybridSearchService:
    """Performs hybrid dense vector search (pgvector) + sparse keyword search (BM25/tsvector) combined via Reciprocal Rank Fusion (RRF)."""

    def __init__(self, db: AsyncSession, tenant_id: uuid.UUID):
        self.db = db
        self.tenant_id = tenant_id

    async def dense_vector_search(
        self, query_vector: List[float], limit: int = 25, document_id: Optional[uuid.UUID] = None
    ) -> List[SearchResult]:
        """Dense semantic search using pgvector cosine distance."""
        cosine_distance = DocumentChunk.embedding.cosine_distance(query_vector).label("distance")
        
        stmt = (
            select(
                DocumentChunk.id,
                DocumentChunk.document_id,
                Document.filename,
                DocumentChunk.page_number,
                DocumentChunk.content,
                cosine_distance,
            )
            .join(Document, DocumentChunk.document_id == Document.id)
            .where(
                DocumentChunk.tenant_id == self.tenant_id,
                Document.status == "indexed",
            )
        )

        if document_id:
            stmt = stmt.where(DocumentChunk.document_id == document_id)

        stmt = stmt.order_by(cosine_distance.asc()).limit(limit)

        result = await self.db.execute(stmt)
        rows = result.all()

        results = []
        for row in rows:
            # Convert cosine distance (0..2) to similarity score (0..1)
            similarity = max(0.0, 1.0 - (row.distance / 2.0))
            results.append(
                SearchResult(
                    chunk_id=row.id,
                    document_id=row.document_id,
                    filename=row.filename,
                    page_number=row.page_number,
                    content=row.content,
                    score=similarity,
                    source="dense",
                )
            )
        return results

    async def sparse_keyword_search(
        self, query_text: str, limit: int = 25, document_id: Optional[uuid.UUID] = None
    ) -> List[SearchResult]:
        """Sparse BM25-style keyword search using PostgreSQL full-text search with websearch_to_tsquery."""
        # Sanitize query for PostgreSQL websearch tsquery
        clean_query = query_text.strip().replace("'", "")
        if not clean_query:
            return []

        ts_query = func.websearch_to_tsquery("english", clean_query)
        ts_rank = func.ts_rank_cd(DocumentChunk.tsv_content, ts_query).label("rank")

        stmt = (
            select(
                DocumentChunk.id,
                DocumentChunk.document_id,
                Document.filename,
                DocumentChunk.page_number,
                DocumentChunk.content,
                ts_rank,
            )
            .join(Document, DocumentChunk.document_id == Document.id)
            .where(
                DocumentChunk.tenant_id == self.tenant_id,
                Document.status == "indexed",
                DocumentChunk.tsv_content.op("@@")(ts_query),
            )
        )

        if document_id:
            stmt = stmt.where(DocumentChunk.document_id == document_id)

        stmt = stmt.order_by(desc("rank")).limit(limit)

        try:
            result = await self.db.execute(stmt)
            rows = result.all()
            return [
                SearchResult(
                    chunk_id=row.id,
                    document_id=row.document_id,
                    filename=row.filename,
                    page_number=row.page_number,
                    content=row.content,
                    score=float(row.rank),
                    source="sparse",
                )
                for row in rows
            ]
        except Exception as e:
            app_logger.warning(f"Keyword search encountered syntax error or empty match: {e}")
            return []

    async def hybrid_search(
        self, query: str, limit: int = 15, document_id: Optional[uuid.UUID] = None, rrf_k: int = 60
    ) -> List[SearchResult]:
        """Combines Dense Vector and Sparse Keyword Search results using Reciprocal Rank Fusion (RRF)."""
        # 1. Generate query embedding
        query_vector = EmbeddingService.generate_query_embedding(query)

        # 2. Run both searches
        dense_results = await self.dense_vector_search(query_vector, limit=limit * 2, document_id=document_id)
        sparse_results = await self.sparse_keyword_search(query, limit=limit * 2, document_id=document_id)

        # 3. Reciprocal Rank Fusion
        rrf_scores: Dict[uuid.UUID, float] = {}
        item_map: Dict[uuid.UUID, SearchResult] = {}

        # Add Dense ranks
        for rank, res in enumerate(dense_results):
            rrf_scores[res.chunk_id] = rrf_scores.get(res.chunk_id, 0.0) + (1.0 / (rrf_k + rank + 1))
            item_map[res.chunk_id] = res

        # Add Sparse ranks
        for rank, res in enumerate(sparse_results):
            rrf_scores[res.chunk_id] = rrf_scores.get(res.chunk_id, 0.0) + (1.0 / (rrf_k + rank + 1))
            if res.chunk_id not in item_map:
                item_map[res.chunk_id] = res

        # 4. Sort by merged RRF score
        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
        final_results: List[SearchResult] = []

        for cid in sorted_chunk_ids[:limit]:
            res = item_map[cid]
            res.score = rrf_scores[cid]
            res.source = "hybrid"
            final_results.append(res)

        return final_results
