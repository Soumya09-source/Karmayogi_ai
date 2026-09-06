import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String

from app.db import Base


class BehaviouralRating(Base):
    __tablename__ = "behavioural_ratings"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))

    employee_id = Column(
        String,
        ForeignKey("employee_profile.employee_id"),
        nullable=False,
        index=True,
    )

    # self or manager
    rater_type = Column(String, nullable=False)

    # Person giving the rating.
    # For self-rating, this is the employee's own ID.
    rater_id = Column(String, nullable=False)

    # Example: communication, teamwork, leadership, adaptability
    competency_area = Column(String, nullable=False)

    # Rating on a 1–5 scale
    rating = Column(Integer, nullable=False)

    timestamp = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )