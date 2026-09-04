import enum

from sqlalchemy import Column, String, Enum

from app.db import Base


class Role(str, enum.Enum):
    employee = "employee"
    trainer = "trainer"
    admin = "admin"


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(Enum(Role), nullable=False)