from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db import get_db
from app.models.recommendation import Recommendation
from app.schemas.recommendation import RecommendationResponse


router = APIRouter(
    prefix="/recommendations",
    tags=["Recommendations"]
)


@router.get(
    "/{employee_id}",
    response_model=list[RecommendationResponse]
)
def get_recommendations(
    employee_id: str,
    db: Session = Depends(get_db)
):
    recommendations = (
        db.query(Recommendation)
        .filter(Recommendation.employee_id == employee_id)
        .order_by(Recommendation.similarity_score.desc())
        .all()
    )

    return recommendations