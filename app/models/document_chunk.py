from sqlalchemy import Column, ForeignKey, Index, Integer, String, Text
from pgvector.sqlalchemy import Vector

from app.db import Base


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    chunk_id = Column(String, primary_key=True)
    parent_doc_id = Column(String, nullable=False)
    chunk_text = Column(Text, nullable=False)
    chunk_order = Column(Integer)
    page_ref = Column(String)
    embedding = Column(Vector(384))

    __table_args__ = (
        Index(
            "idx_document_chunks_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index(
            "idx_document_chunks_parent_doc",
            "parent_doc_id",
        ),
    )


class ChunkDomainTag(Base):
    __tablename__ = "chunk_domain_tags"

    chunk_id = Column(
        String,
        ForeignKey("document_chunks.chunk_id"),
        primary_key=True
    )
    domain = Column(String, primary_key=True)