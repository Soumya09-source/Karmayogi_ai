import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, String, Text

from app.db import Base


class ConceptReviewQueue(Base):
    """
    Concepts extracted from a document_chunk by the LLM that did not
    confidently match any existing row in concept_taxonomy.

    Nothing here is auto-promoted into concept_taxonomy. An admin reviews
    each row and either:
      - maps it to an existing canonical_concept_id (it was a naming
        variant the embedding-similarity match missed), or
      - approves it as a genuinely new concept (a separate, manual step
        of inserting into concept_taxonomy — intentionally not automated,
        since a wrong canonical concept silently corrupts every mastery
        row tagged to it later).
    """

    __tablename__ = "concept_review_queue"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    raw_concept_name = Column(String, nullable=False)
    suggested_domain = Column(String)  # LLM's best-guess domain, not authoritative
    source_chunk_id = Column(String, ForeignKey("document_chunks.chunk_id"), nullable=False)

    # best similarity score found against existing concept_taxonomy rows,
    # for the admin's context on how close a call this was
    best_match_concept_id = Column(String, ForeignKey("concept_taxonomy.canonical_concept_id"))
    best_match_score = Column(Float)

    # pending | approved_mapped_to_existing | approved_new_concept | rejected
    status = Column(String, nullable=False, default="pending")
    resolved_canonical_concept_id = Column(String)  # filled in once an admin resolves it

    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime)
