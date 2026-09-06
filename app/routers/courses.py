import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.deps import require_role
from app.db import get_db
from app.models.course import Course
from app.services.storage_service import upload_file


router = APIRouter(
    prefix="/courses",
    tags=["Courses"],
)


@router.post("/upload")
def upload_course(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: str = Form(""),
    domain_tag: str = Form(""),
    db: Session = Depends(get_db),
    user=Depends(require_role("trainer", "admin")),
):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="A file must be uploaded",
        )

    course_id = f"course_{uuid.uuid4().hex[:12]}"

    object_name = f"{course_id}/{file.filename}"

    upload_file(
        file.file,
        object_name,
        file.content_type or "application/octet-stream",
    )

    course = Course(
        course_id=course_id,
        source_platform="uploaded",
        name=name,
        description=description,
        internal_category=domain_tag,
        source="admin_upload",
        source_url=object_name,
        embedding_text=f"{name} {description}".strip(),
        embedding_text_is_description_based=True,
    )

    db.add(course)
    db.commit()
    db.refresh(course)

    return {
        "message": "Course uploaded successfully",
        "course_id": course.course_id,
        "file_path": object_name,
        "domain_tag": domain_tag,
    }