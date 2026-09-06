from app.celery_app import celery_app
from app.db import SessionLocal
from app.models.document_chunk import DocumentChunk
from app.services.embedding_service import embed_texts_for_app


@celery_app.task
def embed_course_task(course_id: str):
    print(f"Embedding task started for course: {course_id}")

    db = SessionLocal()

    try:
        chunks = (
            db.query(DocumentChunk)
            .filter(
                DocumentChunk.parent_doc_id == course_id,
                DocumentChunk.embedding.is_(None),
            )
            .order_by(DocumentChunk.chunk_order)
            .all()
        )

        if not chunks:
            return {
                "status": "completed",
                "course_id": course_id,
                "message": "No chunks need embedding",
                "embeddings_created": 0,
            }

        texts = [chunk.chunk_text for chunk in chunks]

        embeddings = embed_texts_for_app(texts)

        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding.tolist()

        db.commit()

        return {
            "status": "completed",
            "course_id": course_id,
            "message": "Course embeddings created successfully",
            "embeddings_created": len(embeddings),
        }

    except Exception as e:
        db.rollback()
        raise e

    finally:
        db.close()


@celery_app.task
def generate_mcqs_task(doc_id: str):
    print(f"MCQ generation task started for document: {doc_id}")

    return {
        "status": "completed",
        "doc_id": doc_id,
        "message": "MCQ generation task executed",
    }
@celery_app.task
def notify_high_priority_flag_task(mcq_id: str):
    print(f"NOTIFICATION: MCQ {mcq_id} has been flagged as high priority.")

    return {
        "status": "completed",
        "mcq_id": mcq_id,
        "message": "High-priority MCQ notification sent",
    }
