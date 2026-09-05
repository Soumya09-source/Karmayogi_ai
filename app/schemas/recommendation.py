from datetime import datetime

from pydantic import BaseModel


class RecommendationResponse(BaseModel):
    employee_id: str
    gap_concept_id: str
    recommended_course_id: str
    similarity_score: float
    timestamp: datetime
    status: str

    class Config:
        from_attributes = True