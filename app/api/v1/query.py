from typing import Optional
import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.db.session import get_db
from app.db.models import Tenant
from app.core.security import get_current_tenant
from app.services.hybrid_search import HybridSearchService
from app.services.reranker import RerankerService
from app.services.rag_engine import RAGEngine, RAGResponse

router = APIRouter(prefix="/query", tags=["Query & RAG"])


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=1000, description="User question or inquiry")
    document_id: Optional[uuid.UUID] = Field(
        None, description="Optional document UUID to filter search strictly to a single document"
    )
    top_k: int = Field(5, ge=1, le=15, description="Number of passages to rerank for context")


@router.post("", response_model=RAGResponse)
async def query_documents(
    payload: QueryRequest,
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    """Executes Hybrid Search (Dense + BM25) -> Reranks passages -> Generates strictly cited structured JSON response."""
    # 1. Hybrid Search (Vector + Keyword via RRF)
    search_service = HybridSearchService(db=db, tenant_id=tenant.id)
    candidates = await search_service.hybrid_search(
        query=payload.query,
        limit=payload.top_k * 3,
        document_id=payload.document_id,
    )

    if not candidates:
        return RAGResponse(
            answer="I cannot find this information in the uploaded documents.",
            confidence="low",
            citations=[],
            information_found=False,
        )

    # 2. Rerank top passages (FlashRank / Cohere)
    top_passages = RerankerService.rerank(query=payload.query, candidates=candidates, top_k=payload.top_k)

    # 3. Generate strict cited answer
    response = await RAGEngine.generate_answer(query=payload.query, passages=top_passages)
    return response


@router.post("/stream")
async def query_documents_stream(
    payload: QueryRequest,
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    """Token-by-token Server-Sent Events (SSE) streaming endpoint with final citation payload."""
    search_service = HybridSearchService(db=db, tenant_id=tenant.id)
    candidates = await search_service.hybrid_search(
        query=payload.query,
        limit=payload.top_k * 3,
        document_id=payload.document_id,
    )

    top_passages = RerankerService.rerank(query=payload.query, candidates=candidates, top_k=payload.top_k)

    async def event_generator():
        async for chunk_json in RAGEngine.stream_answer(query=payload.query, passages=top_passages):
            yield {"event": "message", "data": chunk_json}

    return EventSourceResponse(event_generator())
