from sqlalchemy import Column, Integer, String

from app.db import Base


class CompetencyFrameworkMatrix(Base):
    __tablename__ = "competency_framework_matrix"

    designation = Column(String, primary_key=True)
    competency_domain = Column(String, primary_key=True)
    expected_proficiency_level = Column(Integer, nullable=False)  # 1-5
