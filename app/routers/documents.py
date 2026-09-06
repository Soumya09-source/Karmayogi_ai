import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.deps import require_role
from app.db import get_db
from app.models.document_chunk import DocumentChunk
from app.services.storage_service import upload_file
from app.services.text_processing_service import extract_text, chunk_text
from app.tasks.ingestion import embed_course_task


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


@router.post("/upload")
def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user=Depends(require_role("trainer", "admin")),
):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="A file must be uploaded",
        )

    document_id = f"document_{uuid.uuid4().hex[:12]}"

    try:
        extracted_text = extract_text(file.file, file.filename)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    file.file.seek(0)

    object_name = f"{document_id}/{file.filename}"

    upload_file(
        file.file,
        object_name,
        file.content_type or "application/octet-stream",
    )

    chunks = chunk_text(extracted_text)

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No text could be extracted from the uploaded document",
        )

    for index, chunk in enumerate(chunks):
        document_chunk = DocumentChunk(
            chunk_id=f"chunk_{uuid.uuid4().hex[:12]}",
            parent_doc_id=document_id,
            chunk_text=chunk,
            chunk_order=index,
            page_ref=None,
            embedding=None,
        )

        db.add(document_chunk)

    db.commit()

    task = embed_course_task.delay(document_id)

    return {
        "message": "Document uploaded and embedding task queued successfully",
        "document_id": document_id,
        "filename": file.filename,
        "file_path": object_name,
        "chunks_created": len(chunks),
        "task_id": task.id,
        "task_status": "queued",
        "uploaded_by": user.id,
    }