from datetime import datetime

from sqlalchemy import Column, DateTime, Float, String

from app.db import Base


class Recommendation(Base):
    __tablename__ = "recommendations"

    employee_id = Column(String, primary_key=True)
    gap_concept_id = Column(String, primary_key=True)
    recommended_course_id = Column(String, primary_key=True)

    similarity_score = Column(Float, nullable=False)

    timestamp = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )

    status = Column(
        String,
        default="active",
        nullable=False
    )