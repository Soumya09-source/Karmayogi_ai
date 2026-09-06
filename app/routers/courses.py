import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.deps import require_role
from app.db import get_db
from app.models.course import Course
from app.models.document_chunk import DocumentChunk
from app.services.storage_service import upload_file
from app.services.text_processing_service import extract_text, chunk_text
from app.tasks.ingestion import embed_course_task


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

    try:
        # Extract text from uploaded file
        extracted_text = extract_text(file.file, file.filename)

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    # Return to the beginning so MinIO can upload the complete file
    file.file.seek(0)

    object_name = f"{course_id}/{file.filename}"

    # Upload original file to MinIO
    upload_file(
        file.file,
        object_name,
        file.content_type or "application/octet-stream",
    )

    # Create course record
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

    # Split extracted text into chunks
    chunks = chunk_text(extracted_text)

    # Save chunks WITHOUT embeddings
    for index, chunk in enumerate(chunks):
        document_chunk = DocumentChunk(
            chunk_id=f"chunk_{uuid.uuid4().hex[:12]}",
            parent_doc_id=course_id,
            chunk_text=chunk,
            chunk_order=index,
            page_ref=None,
            embedding=None,
        )

        db.add(document_chunk)

    # Save course and chunks first
    db.commit()
    db.refresh(course)

    # Queue embedding task in Celery
    task = embed_course_task.delay(course_id)

    return {
        "message": "Course uploaded and embedding task queued successfully",
        "course_id": course.course_id,
        "file_path": object_name,
        "domain_tag": domain_tag,
        "chunks_created": len(chunks),
        "task_id": task.id,
        "task_status": "queued",
    }