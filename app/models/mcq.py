import uuid

from sqlalchemy import Column, ForeignKey, Integer, String
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
    options = Column(JSONB, nullable=False)  # e.g. [{"id": "a", "text": "..."}, ...]
    correct_option_id = Column(String, nullable=False)
    difficulty = Column(String, nullable=False)  # easy | medium | hard

    # live | flagged | flagged_high_priority | rejected
    # Only status == "live" rows are ever eligible to be served in a quiz.
    status = Column(String, nullable=False, default="live")

    times_served = Column(Integer, nullable=False, default=0)
    times_correct = Column(Integer, nullable=False, default=0)
