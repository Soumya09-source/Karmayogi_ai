from datetime import datetime

from sqlalchemy import Column, DateTime, String

from app.db import Base


class Document(Base):
    __tablename__ = "documents"

    document_id = Column(String, primary_key=True)
    filename = Column(String, nullable=False)
    uploaded_by = Column(String, nullable=False)
    status = Column(String, nullable=False, default="sanity_pending")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    approved_at = Column(DateTime, nullable=True)