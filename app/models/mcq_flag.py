import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, String

from app.db import Base


class MCQFlag(Base):
    __tablename__ = "mcq_flags"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    mcq_id = Column(String, ForeignKey("mcqs.id"), nullable=False)
    flagged_by = Column(String, nullable=False)
    reason = Column(String, nullable=False)
    comment = Column(String, nullable=True)
    session_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
