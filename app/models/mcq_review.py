import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, JSON, String

from app.db import Base


class MCQReview(Base):
    __tablename__ = "mcq_reviews"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    mcq_id = Column(String, ForeignKey("mcqs.id"), nullable=False)
    reviewed_by = Column(String, nullable=False)
    decision = Column(String, nullable=False)
    pre_edit_snapshot = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
