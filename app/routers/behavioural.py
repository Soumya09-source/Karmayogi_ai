from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.deps import require_role, get_current_user
from app.db import get_db
from app.models.behavioural_rating import BehaviouralRating
from app.models.user import User
from app.schemas.behavioural import (
    ManagerRatingRequest,
    SelfRatingRequest,
)

router = APIRouter(
    prefix="/behavioural",
    tags=["behavioural"],
)


@router.post("/self-rating")
def submit_self_rating(
    request: SelfRatingRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("employee")),
):
    if current_user.id != request.employee_id:
        raise HTTPException(
            status_code=403,
            detail="Employees can submit self-ratings only for themselves",
        )

    rating = BehaviouralRating(
        employee_id=current_user.id,
        rater_type="self",
        rater_id=current_user.id,
        competency_area=request.competency_area,
        rating=request.rating,
    )

    db.add(rating)
    db.commit()
    db.refresh(rating)

    return {
        "id": rating.id,
        "employee_id": rating.employee_id,
        "rater_type": rating.rater_type,
        "competency_area": rating.competency_area,
        "rating": rating.rating,
        "timestamp": rating.timestamp,
    }


@router.post("/manager-rating")
def submit_manager_rating(
    request: ManagerRatingRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("trainer", "admin")),
):
    rating = BehaviouralRating(
        employee_id=request.employee_id,
        rater_type="manager",
        rater_id=current_user.id,
        competency_area=request.competency_area,
        rating=request.rating,
    )

    db.add(rating)
    db.commit()
    db.refresh(rating)

    return {
        "id": rating.id,
        "employee_id": rating.employee_id,
        "rater_type": rating.rater_type,
        "rater_id": rating.rater_id,
        "competency_area": rating.competency_area,
        "rating": rating.rating,
        "timestamp": rating.timestamp,
    }


@router.get("/{employee_id}")
def get_behavioural_ratings(
    employee_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if (
        current_user.role.value == "employee"
        and current_user.id != employee_id
    ):
        raise HTTPException(
            status_code=403,
            detail="Employees can access only their own behavioural ratings",
        )

    ratings = (
        db.query(BehaviouralRating)
        .filter(BehaviouralRating.employee_id == employee_id)
        .order_by(BehaviouralRating.timestamp.desc())
        .all()
    )

    return [
        {
            "id": rating.id,
            "employee_id": rating.employee_id,
            "rater_type": rating.rater_type,
            "rater_id": rating.rater_id,
            "competency_area": rating.competency_area,
            "rating": rating.rating,
            "timestamp": rating.timestamp,
        }
        for rating in ratings
    ]