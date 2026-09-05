from sqlalchemy import Column, ForeignKey, String

from app.db import Base


class EmployeeTrainingHistory(Base):
    __tablename__ = "employee_training_history"

    employee_id = Column(
        String,
        ForeignKey("employee_profile.employee_id"),
        primary_key=True,
    )
    course_id = Column(
        String,
        ForeignKey("courses.course_id"),
        primary_key=True,
    )
