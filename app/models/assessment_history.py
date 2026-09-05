import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, String

from app.db import Base


class AssessmentHistory(Base):
    """Immutable log of every answered question. Never updated in place."""

    __tablename__ = "assessment_history"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, nullable=False, index=True)
    employee_id = Column(
        String,
        ForeignKey("employee_profile.employee_id"),
        nullable=False,
    )
    mcq_id = Column(String, ForeignKey("mcqs.id"), nullable=False)
    concept_id = Column(
        String,
        ForeignKey("concept_taxonomy.canonical_concept_id"),
        nullable=False,
    )
    correct = Column(Boolean, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
