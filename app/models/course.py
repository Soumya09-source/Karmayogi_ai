from sqlalchemy import Boolean, Column, Index, Integer, Numeric, String, Text
from pgvector.sqlalchemy import Vector

from app.db import Base


class Course(Base):
    __tablename__ = "courses"

    course_id = Column(String, primary_key=True)
    source_platform = Column(String)
    name = Column(String, nullable=False)
    description = Column(Text)
    provider_organization = Column(String)
    level = Column(String)
    duration_minutes = Column(Numeric)
    start_date = Column(String)
    end_date = Column(String)
    enrollment_type = Column(String)
    status = Column(Integer)
    course_url = Column(String)
    internal_category = Column(String)
    source = Column(String)
    source_url = Column(String)
    embedding_text = Column(Text)
    embedding_text_is_description_based = Column(Boolean)
    embedding = Column(Vector(384))
    __table_args__ = (
        Index(
            "idx_courses_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )