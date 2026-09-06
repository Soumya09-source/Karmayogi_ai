from pydantic import BaseModel, Field


class SelfRatingRequest(BaseModel):
    employee_id: str
    competency_area: str
    rating: int = Field(ge=1, le=5)


class ManagerRatingRequest(BaseModel):
    employee_id: str
    rater_id: str
    competency_area: str
    rating: int = Field(ge=1, le=5)