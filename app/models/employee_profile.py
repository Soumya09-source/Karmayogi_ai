from sqlalchemy import Column, Integer, String

from app.db import Base


class EmployeeProfile(Base):
    __tablename__ = "employee_profile"

    employee_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    designation = Column(String, nullable=False)
    years_of_service = Column(Integer)
    department = Column(String)
