from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.deps import require_role
from app.db import get_db
from app.models.mcq import MCQ
from app.models.mcq_flag import MCQFlag
from app.models.user import User
from app.schemas.mcq_flag import MCQFlagCreate
from app.schemas.mcq_review import MCQReviewCreate
from app.models.mcq_review import MCQReview
from app.tasks.ingestion import notify_high_priority_flag_task


router = APIRouter(
    prefix="/mcqs",
    tags=["MCQs"],
)


@router.post("/{mcq_id}/flag")
def flag_mcq(
    mcq_id: str,
    flag_data: MCQFlagCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("employee")),
):
    mcq = db.query(MCQ).filter(MCQ.id == mcq_id).first()

    if mcq is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="MCQ not found",
        )

    flag = MCQFlag(
        mcq_id=mcq.id,
        flagged_by=current_user.id,
        reason=flag_data.reason,
        comment=flag_data.comment,
        session_id=flag_data.session_id,
    )

    db.add(flag)

    mcq.flag_count += 1

    if mcq.flag_count >= 3:
        mcq.status = "flagged_high_priority"
        notify_high_priority_flag_task.delay(mcq.id)
    else:
        mcq.status = "flagged"

    db.commit()
    db.refresh(mcq)

    return {
        "message": "MCQ flagged successfully",
        "mcq_id": mcq.id,
        "flag_count": mcq.flag_count,
        "status": mcq.status,
    }

@router.get("/review-queue")
def get_review_queue(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("trainer", "admin")),
):
    mcqs = (
        db.query(MCQ)
        .filter(MCQ.status.in_(["flagged", "flagged_high_priority"]))
        .order_by(
            MCQ.status.desc(),
            MCQ.flag_count.desc(),
            MCQ.id.asc(),
        )
        .all()
    )

    return {
        "count": len(mcqs),
        "items": [
            {
                "id": mcq.id,
                "concept_id": mcq.concept_id,
                "options": mcq.options,
                "correct_option_id": mcq.correct_option_id,
                "difficulty": mcq.difficulty,
                "status": mcq.status,
                "flag_count": mcq.flag_count,
                "explanation": mcq.explanation,
            }
            for mcq in mcqs
        ],
    }


@router.post("/{mcq_id}/review")
def review_mcq(
    mcq_id: str,
    review_data: MCQReviewCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("trainer", "admin")),
):
    mcq = db.query(MCQ).filter(MCQ.id == mcq_id).first()

    if mcq is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="MCQ not found",
        )

    pre_edit_snapshot = None

    if review_data.decision == "confirmed_bad":
        mcq.status = "rejected"

    elif review_data.decision == "confirmed_correct":
        mcq.status = "live"
        mcq.flag_count = 0

    elif review_data.decision == "edited":
        pre_edit_snapshot = {
            "options": mcq.options,
            "correct_option_id": mcq.correct_option_id,
            "difficulty": mcq.difficulty,
            "explanation": mcq.explanation,
        }

        if review_data.options is not None:
            mcq.options = review_data.options

        if review_data.correct_option_id is not None:
            mcq.correct_option_id = review_data.correct_option_id

        if review_data.difficulty is not None:
            mcq.difficulty = review_data.difficulty

        if review_data.explanation is not None:
            mcq.explanation = review_data.explanation

        mcq.status = "live"
        mcq.flag_count = 0

    review = MCQReview(
        mcq_id=mcq.id,
        reviewed_by=current_user.id,
        decision=review_data.decision,
        pre_edit_snapshot=pre_edit_snapshot,
    )

    db.add(review)
    db.commit()
    db.refresh(mcq)

    return {
        "message": "MCQ reviewed successfully",
        "mcq_id": mcq.id,
        "decision": review_data.decision,
        "status": mcq.status,
        "flag_count": mcq.flag_count,
    }
