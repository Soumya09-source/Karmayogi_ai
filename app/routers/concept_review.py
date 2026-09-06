import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.deps import require_role
from app.db import get_db
from app.models.concept import ConceptTaxonomy
from app.models.concept_review_queue import ConceptReviewQueue


router = APIRouter(
    prefix="/concept-review",
    tags=["Concept Review"],
)


@router.get("/queue")
def get_review_queue(
    db: Session = Depends(get_db),
    user=Depends(require_role("admin")),
):
    items = (
        db.query(ConceptReviewQueue)
        .filter(ConceptReviewQueue.status == "pending")
        .all()
    )

    return items


@router.post("/{queue_id}/approve")
def approve_concept(
    queue_id: str,
    canonical_name: str,
    domain: str,
    db: Session = Depends(get_db),
    user=Depends(require_role("admin")),
):
    item = (
        db.query(ConceptReviewQueue)
        .filter(ConceptReviewQueue.id == queue_id)
        .first()
    )

    if not item:
        raise HTTPException(
            status_code=404,
            detail="Concept review item not found",
        )

    if item.status != "pending":
        raise HTTPException(
            status_code=400,
            detail=f"This item has already been resolved with status: {item.status}",
        )

    concept_id = f"CC-{uuid.uuid4().hex[:8]}"

    new_concept = ConceptTaxonomy(
        canonical_concept_id=concept_id,
        canonical_concept_name=canonical_name,
        raw_concept_name=item.raw_concept_name,
        parent_domain=domain,
        embedding_text=canonical_name,
    )

    db.add(new_concept)

    item.status = "approved_new_concept"
    item.resolved_canonical_concept_id = concept_id
    item.resolved_at = datetime.utcnow()

    db.commit()
    db.refresh(new_concept)

    return {
        "message": "Concept approved and added to taxonomy",
        "concept_id": concept_id,
        "canonical_name": canonical_name,
    }


@router.post("/{queue_id}/reject")
def reject_concept(
    queue_id: str,
    db: Session = Depends(get_db),
    user=Depends(require_role("admin")),
):
    item = (
        db.query(ConceptReviewQueue)
        .filter(ConceptReviewQueue.id == queue_id)
        .first()
    )

    if not item:
        raise HTTPException(
            status_code=404,
            detail="Concept review item not found",
        )

    if item.status != "pending":
        raise HTTPException(
            status_code=400,
            detail=f"This item has already been resolved with status: {item.status}",
        )

    item.status = "rejected"
    item.resolved_at = datetime.utcnow()

    db.commit()

    return {
        "message": "Concept rejected successfully",
        "queue_id": queue_id,
    }