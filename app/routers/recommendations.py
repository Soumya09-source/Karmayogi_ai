from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.deps import get_current_user
from app.db import get_db
from app.models.recommendation import Recommendation
from app.models.concept_mastery import ConceptMastery
from app.models.behavioural_rating import BehaviouralRating
from app.models.user import User
from app.schemas.recommendation import RecommendationResponse
from app.services.recommendation_service import generate_recommendations


router = APIRouter(
    prefix="/recommendations",
    tags=["Recommendations"]
)


def verify_employee_access(employee_id: str, current_user: User):
    if (
        current_user.role.value == "employee"
        and current_user.id != employee_id
    ):
        raise HTTPException(
            status_code=403,
            detail="Employees can access only their own recommendation data",
        )


@router.get("/skill-gap/{employee_id}")
def get_skill_gap(
    employee_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    verify_employee_access(employee_id, current_user)

    technical_gaps = (
        db.query(ConceptMastery)
        .filter(
            ConceptMastery.employee_id == employee_id,
            ConceptMastery.p_mastery_current < 0.7
        )
        .all()
    )

    behavioural_ratings = (
        db.query(BehaviouralRating)
        .filter(
            BehaviouralRating.employee_id == employee_id
        )
        .all()
    )

    return {
        "employee_id": employee_id,
        "technical_gaps": technical_gaps,
        "behavioural_ratings": behavioural_ratings
    }


@router.post(
    "/{employee_id}/{course_id}/status",
    response_model=RecommendationResponse
)
def update_recommendation_status(
    employee_id: str,
    course_id: str,
    status: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    verify_employee_access(employee_id, current_user)

    valid_statuses = {"shown", "enrolled", "ignored"}

    if status not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status. Allowed values: {', '.join(valid_statuses)}"
        )

    recommendation = (
        db.query(Recommendation)
        .filter(
            Recommendation.employee_id == employee_id,
            Recommendation.recommended_course_id == course_id
        )
        .first()
    )

    if not recommendation:
        raise HTTPException(
            status_code=404,
            detail="Recommendation not found"
        )

    recommendation.status = status

    db.commit()
    db.refresh(recommendation)

    return recommendation


@router.get(
    "/{employee_id}",
    response_model=list[RecommendationResponse]
)
def get_recommendations(
    employee_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    verify_employee_access(employee_id, current_user)

    recommendations = (
        db.query(Recommendation)
        .filter(Recommendation.employee_id == employee_id)
        .order_by(Recommendation.similarity_score.desc())
        .all()
    )

    # Generate recommendations automatically if none exist
    if not recommendations:
        recommendations = generate_recommendations(
            db=db,
            employee_id=employee_id
        )

    # Mark recommendations as shown when fetched
    for recommendation in recommendations:
        if recommendation.status == "active":
            recommendation.status = "shown"

    db.commit()

    return recommendations