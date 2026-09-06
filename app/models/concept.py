from sqlalchemy import Boolean, Column, Index, String, Text
from pgvector.sqlalchemy import Vector

from app.db import Base


class ConceptTaxonomy(Base):
    __tablename__ = "concept_taxonomy"

    canonical_concept_id = Column(String, primary_key=True)
    canonical_concept_name = Column(String, nullable=False)
    raw_concept_name = Column(String)
    parent_domain = Column(String)
    competency_area = Column(String)
    source_file = Column(String)
    source_id = Column(String)
    alias_name = Column(String)
    is_canonical_label = Column(Boolean)
    embedding_text = Column(Text)
    embedding = Column(Vector(384))
    __table_args__ = (
        Index(
            "idx_concept_taxonomy_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )