import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db
from app.models.mcq import MCQ
from app.services import bkt

router = APIRouter(prefix="/assessments", tags=["assessments"])


@router.get("/next-question/{employee_id}")
def next_question(employee_id: str, db: Session = Depends(get_db)):
    eligible_concepts = bkt.get_eligible_concepts(db, employee_id)

    if not eligible_concepts:
        raise HTTPException(
            status_code=404,
            detail="No eligible concepts found for this employee",
        )

    concept_id = bkt.select_next_concept_for_employee(
        db,
        employee_id,
        eligible_concepts,
    )

    question = bkt.select_question(
        db,
        concept_id,
        employee_id,
        set(),
    )

    if question is None:
        raise HTTPException(
            status_code=404,
            detail="No question available for the selected concept",
        )

    # Never expose the correct answer to the frontend.
    question.pop("correct_option_id", None)

    return question


@router.post("/answer")
def submit_answer(
    employee_id: str,
    mcq_id: str,
    selected_option_id: str,
    db: Session = Depends(get_db),
):
    # Find the MCQ in the database.
    mcq = db.query(MCQ).filter(MCQ.id == mcq_id).first()

    if mcq is None:
        raise HTTPException(
            status_code=404,
            detail="MCQ not found",
        )

    # Check whether the employee selected the correct option.
    correct = selected_option_id == mcq.correct_option_id

    # Build the MCQ dictionary expected by the BKT service.
    mcq_data = {
        "id": mcq.id,
        "concept_id": mcq.concept_id,
        "options": mcq.options,
        "correct_option_id": mcq.correct_option_id,
        "difficulty": mcq.difficulty,
    }

    # Generate an ID for this API assessment record.
    session_id = str(uuid.uuid4())

    # Record the answer in assessment history.
    bkt.log_assessment_history(
        db,
        session_id,
        employee_id,
        mcq.id,
        mcq.concept_id,
        correct,
    )

    # Apply the Bayesian Knowledge Tracing update.
    new_mastery = bkt.apply_answer(
        db,
        employee_id,
        mcq.concept_id,
        mcq_data,
        correct,
    )

    return {
        "employee_id": employee_id,
        "mcq_id": mcq.id,
        "correct": correct,
        "mastery": new_mastery,
    }