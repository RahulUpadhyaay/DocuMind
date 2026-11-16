import base64
import uuid
from typing import Optional
from app.workers.celery_app import celery_app
from app.db.session import SyncSessionLocal
from app.db.models import Document, DocumentChunk
from app.services.parser import DocumentParser
from app.services.chunker import DocumentChunker
from app.services.embeddings import EmbeddingService
from app.core.logging import app_logger


def execute_document_ingestion(document_id_str: str, file_bytes: bytes, filename: str):
    """Core ingestion logic executed synchronously by Celery worker or FastAPI BackgroundTasks."""
    doc_uuid = uuid.UUID(document_id_str)
    
    with SyncSessionLocal() as session:
        doc = session.query(Document).filter(Document.id == doc_uuid).first()
        if not doc:
            app_logger.error(f"Document {document_id_str} not found in database.")
            return

        try:
            # 1. Update status to processing
            doc.status = "processing"
            session.commit()

            # 2. Extract text page-by-page
            pages = DocumentParser.parse(filename, file_bytes)
            if not pages:
                raise ValueError("No extractable text found in the document.")

            doc.total_pages = len(pages)

            # 3. Chunk text keeping page references
            chunker = DocumentChunker()
            chunks = chunker.chunk_pages(pages)

            # 4. Generate dense embeddings in batches
            texts = [c.content for c in chunks]
            embeddings = EmbeddingService.generate_embeddings(texts)

            # 5. Insert document chunks into database
            chunk_records = []
            for chunk_obj, embedding in zip(chunks, embeddings):
                chunk_rec = DocumentChunk(
                    id=uuid.uuid4(),
                    tenant_id=doc.tenant_id,
                    document_id=doc.id,
                    chunk_index=chunk_obj.chunk_index,
                    page_number=chunk_obj.page_number,
                    content=chunk_obj.content,
                    embedding=embedding,
                    chunk_metadata=chunk_obj.metadata,
                )
                chunk_records.append(chunk_rec)

            session.add_all(chunk_records)

            # 6. Mark document as indexed
            doc.status = "indexed"
            doc.error_message = None
            session.commit()
            app_logger.info(
                f"Successfully indexed document {doc.filename} ({doc.id}): {len(pages)} pages, {len(chunk_records)} chunks."
            )

        except Exception as e:
            session.rollback()
            doc.status = "failed"
            doc.error_message = str(e)
            session.commit()
            app_logger.error(f"Failed to process document {document_id_str}: {e}")
            raise


@celery_app.task(name="app.workers.tasks.process_document_celery", bind=True, max_retries=2)
def process_document_celery(self, document_id_str: str, file_b64: str, filename: str):
    """Celery task entry point."""
    try:
        file_bytes = base64.b64decode(file_b64)
        execute_document_ingestion(document_id_str, file_bytes, filename)
    except Exception as exc:
        app_logger.error(f"Celery task retry for document {document_id_str}: {exc}")
        raise self.retry(exc=exc, countdown=10)
