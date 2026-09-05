from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String

from app.db import Base


class ConceptMastery(Base):
    """
    Per (employee, concept) BKT state.

    Note: p_g is stored for completeness / future personalization but is
    NOT what bayesian_update() uses in practice — the guess probability is
    always derived as 1/len(mcq.options) at answer time, per concept
    (never hardcoded). p_s IS used directly from this row (falls back to
    0.1 if not yet personalized). p_t is only ever applied at session
    start (see bkt.begin_session), never mid-session.
    """

    __tablename__ = "concept_mastery"

    employee_id = Column(
        String,
        ForeignKey("employee_profile.employee_id"),
        primary_key=True,
    )
    concept_id = Column(
        String,
        ForeignKey("concept_taxonomy.canonical_concept_id"),
        primary_key=True,
    )

    p_l0 = Column(Float, nullable=False)
    p_t = Column(Float, nullable=False, default=0.15)
    p_g = Column(Float, nullable=False, default=0.25)
    p_s = Column(Float, nullable=False, default=0.1)
    p_mastery_current = Column(Float, nullable=False)

    attempt_count = Column(Integer, nullable=False, default=0)
    last_updated = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
