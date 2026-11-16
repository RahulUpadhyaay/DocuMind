import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    String,
    Integer,
    BigInteger,
    Text,
    DateTime,
    ForeignKey,
    Boolean,
    Index,
    Computed,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, TSVECTOR
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

from app.db.base import Base
from app.core.config import settings


class Tenant(Base):
    __tablename__ = "tenants"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    api_key_hash = Column(String(255), nullable=False, unique=True, index=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    documents = relationship("Document", back_populates="tenant", cascade="all, delete-orphan")
    chunks = relationship("DocumentChunk", back_populates="tenant", cascade="all, delete-orphan")


class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)
    filename = Column(String(512), nullable=False)
    file_size_bytes = Column(BigInteger, nullable=False)
    content_hash = Column(String(64), nullable=False, index=True)
    status = Column(String(50), default="pending", nullable=False)  # pending, processing, indexed, failed
    total_pages = Column(Integer, default=0, nullable=False)
    error_message = Column(Text, nullable=True)
    doc_metadata = Column("metadata", JSONB, default=dict)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    tenant = relationship("Tenant", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    page_number = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    
    # Dense vector embedding
    embedding = Column(Vector(settings.EMBEDDING_DIMENSION), nullable=True)
    
    # Sparse BM25 / tsvector generated column
    tsv_content = Column(
        TSVECTOR,
        Computed("to_tsvector('english', content)", persisted=True),
        nullable=True
    )
    
    chunk_metadata = Column("metadata", JSONB, default=dict)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    tenant = relationship("Tenant", back_populates="chunks")
    document = relationship("Document", back_populates="chunks")

    __table_args__ = (
        Index("idx_chunks_tenant_doc", "tenant_id", "document_id"),
        Index("idx_chunks_page", "document_id", "page_number"),
        Index("idx_chunks_tsv", "tsv_content", postgresql_using="gin"),
    )
