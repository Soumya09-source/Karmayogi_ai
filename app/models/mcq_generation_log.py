from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text

from app.db import Base


class McqGenerationLog(Base):
    """
    Tracks MCQ-generation progress per chunk, deliberately kept separate
    from `document_chunks` (owned/actively changed on the embedding-pipeline
    branch) rather than adding a status column there — avoids a repeat of
    the schema-collision issues hit during BKT integration.
    """

    __tablename__ = "mcq_generation_log"

    chunk_id = Column(String, primary_key=True)  # FK to document_chunks.chunk_id,
                                                   # not declared as a hard FK so this
                                                   # table's migration doesn't depend
                                                   # on exact timing of that branch's merges
    status = Column(String, nullable=False, default="pending")
    # pending | processed | skipped_no_concepts | error

    concepts_found = Column(Integer, default=0)
    mcqs_generated = Column(Integer, default=0)
    error_message = Column(Text)
    processed_at = Column(DateTime)
