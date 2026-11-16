import base64
import hashlib
import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from pydantic import BaseModel

from app.db.session import get_db
from app.db.models import Tenant, Document, DocumentChunk
from app.core.security import get_current_tenant
from app.core.config import settings
from app.core.logging import app_logger
from app.workers.tasks import process_document_celery, execute_document_ingestion

router = APIRouter(prefix="/documents", tags=["Documents"])


class DocumentResponse(BaseModel):
    id: uuid.UUID
    filename: str
    file_size_bytes: int
    status: str
    total_pages: int
    error_message: Optional[str] = None
    created_at: str


class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int


@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    """Uploads a document (PDF, DOCX, TXT), registers it under the tenant, and queues background parsing & vectorization."""
    # Validate extension
    filename = file.filename or "unknown.txt"
    lower_name = filename.lower()
    if not lower_name.endswith((".pdf", ".docx", ".txt", ".md")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported format. Only PDF, DOCX, and TXT are supported.",
        )

    file_bytes = await file.read()
    file_size = len(file_bytes)

    # Size limit check (25MB default)
    if file_size > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum upload limit of {settings.MAX_UPLOAD_SIZE_MB}MB.",
        )

    # Compute SHA-256 hash for deduplication per tenant
    content_hash = hashlib.sha256(file_bytes).hexdigest()

    # Check if duplicate exists for this tenant
    stmt = select(Document).where(Document.tenant_id == tenant.id, Document.content_hash == content_hash)
    result = await db.execute(stmt)
    existing_doc = result.scalar_one_or_none()
    if existing_doc:
        return DocumentResponse(
            id=existing_doc.id,
            filename=existing_doc.filename,
            file_size_bytes=existing_doc.file_size_bytes,
            status=existing_doc.status,
            total_pages=existing_doc.total_pages,
            error_message=existing_doc.error_message,
            created_at=existing_doc.created_at.isoformat(),
        )

    # Create new document record
    new_doc = Document(
        tenant_id=tenant.id,
        filename=filename,
        file_size_bytes=file_size,
        content_hash=content_hash,
        status="pending",
        total_pages=0,
    )
    db.add(new_doc)
    await db.commit()
    await db.refresh(new_doc)

    # Enqueue task: Try Celery first, fallback to FastAPI BackgroundTasks if Redis is offline
    dispatched_to_celery = False
    try:
        file_b64 = base64.b64encode(file_bytes).decode("utf-8")
        process_document_celery.delay(str(new_doc.id), file_b64, filename)
        dispatched_to_celery = True
        app_logger.info(f"Enqueued document {new_doc.id} to Celery worker.")
    except Exception as e:
        app_logger.warning(f"Could not connect to Celery/Redis ({e}). Executing with FastAPI BackgroundTasks.")
        background_tasks.add_task(execute_document_ingestion, str(new_doc.id), file_bytes, filename)

    return DocumentResponse(
        id=new_doc.id,
        filename=new_doc.filename,
        file_size_bytes=new_doc.file_size_bytes,
        status="pending" if dispatched_to_celery else "processing",
        total_pages=0,
        error_message=None,
        created_at=new_doc.created_at.isoformat(),
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    """Lists all documents belonging to the authenticated tenant."""
    stmt = select(Document).where(Document.tenant_id == tenant.id).order_by(Document.created_at.desc())
    result = await db.execute(stmt)
    docs = result.scalars().all()

    return DocumentListResponse(
        documents=[
            DocumentResponse(
                id=d.id,
                filename=d.filename,
                file_size_bytes=d.file_size_bytes,
                status=d.status,
                total_pages=d.total_pages,
                error_message=d.error_message,
                created_at=d.created_at.isoformat(),
            )
            for d in docs
        ],
        total=len(docs),
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: uuid.UUID,
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    """Gets status and metadata of a specific document."""
    stmt = select(Document).where(Document.id == document_id, Document.tenant_id == tenant.id)
    result = await db.execute(stmt)
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    return DocumentResponse(
        id=doc.id,
        filename=doc.filename,
        file_size_bytes=doc.file_size_bytes,
        status=doc.status,
        total_pages=doc.total_pages,
        error_message=doc.error_message,
        created_at=doc.created_at.isoformat(),
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: uuid.UUID,
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    """Deletes a document and cascades deletion to all associated chunks/embeddings."""
    stmt = select(Document).where(Document.id == document_id, Document.tenant_id == tenant.id)
    result = await db.execute(stmt)
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    await db.delete(doc)
    await db.commit()
    return None
