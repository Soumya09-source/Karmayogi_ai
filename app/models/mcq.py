import uuid

from sqlalchemy import Column, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB

from app.db import Base


class MCQ(Base):
    __tablename__ = "mcqs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    concept_id = Column(
        String,
        ForeignKey("concept_taxonomy.canonical_concept_id"),
        nullable=False,
    )
    source_chunk_id = Column(String, ForeignKey("document_chunks.chunk_id"))
    options = Column(JSONB, nullable=False)  # e.g. [{"id": "a", "text": "..."}, ...]
    correct_option_id = Column(String, nullable=False)
    explanation = Column(Text)
    difficulty = Column(String, nullable=False)  # easy | medium | hard

    # live | flagged | flagged_high_priority | rejected
    # Only status == "live" rows are ever eligible to be served in a quiz.
    status = Column(String, nullable=False, default="live")

    # From the generation-time self-consistency check (independent
    # re-derivation of the answer from source_chunk_id, compared against
    # correct_option_id). Informational only — does NOT gate publishing,
    # per the reactive-flagging design (trust is verified after the fact,
    # not before). Low-confidence rows are simply worth prioritizing in
    # the trainer review queue once that's built.
    confidence_score = Column(Float)

    times_served = Column(Integer, nullable=False, default=0)
    times_correct = Column(Integer, nullable=False, default=0)
