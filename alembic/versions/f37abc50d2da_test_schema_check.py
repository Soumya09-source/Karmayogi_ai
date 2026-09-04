"""test schema check

Revision ID: f37abc50d2da
Revises: c47aac120b41
Create Date: 2026-09-04 21:49:45.082553

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import pgvector.sqlalchemy


# revision identifiers, used by Alembic.
revision: str = "f37abc50d2da"
down_revision: Union[str, Sequence[str], None] = "c47aac120b41"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""

    op.create_table(
        "concept_taxonomy",
        sa.Column("canonical_concept_id", sa.String(), nullable=False),
        sa.Column("canonical_concept_name", sa.String(), nullable=False),
        sa.Column("raw_concept_name", sa.String(), nullable=True),
        sa.Column("parent_domain", sa.String(), nullable=True),
        sa.Column("competency_area", sa.String(), nullable=True),
        sa.Column("source_file", sa.String(), nullable=True),
        sa.Column("source_id", sa.String(), nullable=True),
        sa.Column("alias_name", sa.String(), nullable=True),
        sa.Column("is_canonical_label", sa.Boolean(), nullable=True),
        sa.Column("embedding_text", sa.Text(), nullable=True),
        sa.Column(
            "embedding",
            pgvector.sqlalchemy.vector.VECTOR(dim=384),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("canonical_concept_id"),
    )

    op.create_table(
        "courses",
        sa.Column("course_id", sa.String(), nullable=False),
        sa.Column("source_platform", sa.String(), nullable=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("provider_organization", sa.String(), nullable=True),
        sa.Column("level", sa.String(), nullable=True),
        sa.Column("duration_minutes", sa.Numeric(), nullable=True),
        sa.Column("start_date", sa.String(), nullable=True),
        sa.Column("end_date", sa.String(), nullable=True),
        sa.Column("enrollment_type", sa.String(), nullable=True),
        sa.Column("status", sa.Integer(), nullable=True),
        sa.Column("course_url", sa.String(), nullable=True),
        sa.Column("internal_category", sa.String(), nullable=True),
        sa.Column("source", sa.String(), nullable=True),
        sa.Column("source_url", sa.String(), nullable=True),
        sa.Column("embedding_text", sa.Text(), nullable=True),
        sa.Column(
            "embedding_text_is_description_based",
            sa.Boolean(),
            nullable=True,
        ),
        sa.Column(
            "embedding",
            pgvector.sqlalchemy.vector.VECTOR(dim=384),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("course_id"),
    )

    op.create_table(
        "document_chunks",
        sa.Column("chunk_id", sa.String(), nullable=False),
        sa.Column("parent_doc_id", sa.String(), nullable=False),
        sa.Column("chunk_text", sa.Text(), nullable=False),
        sa.Column("chunk_order", sa.Integer(), nullable=True),
        sa.Column("page_ref", sa.String(), nullable=True),
        sa.Column(
            "embedding",
            pgvector.sqlalchemy.vector.VECTOR(dim=384),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("chunk_id"),
    )

    op.create_table(
        "chunk_domain_tags",
        sa.Column("chunk_id", sa.String(), nullable=False),
        sa.Column("domain", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(
            ["chunk_id"],
            ["document_chunks.chunk_id"],
        ),
        sa.PrimaryKeyConstraint("chunk_id", "domain"),
    )

    op.execute(
        "CREATE INDEX idx_concept_taxonomy_embedding "
        "ON concept_taxonomy USING hnsw "
        "(embedding vector_cosine_ops)"
    )

    op.execute(
        "CREATE INDEX idx_courses_embedding "
        "ON courses USING hnsw "
        "(embedding vector_cosine_ops)"
    )

    op.execute(
        "CREATE INDEX idx_document_chunks_embedding "
        "ON document_chunks USING hnsw "
        "(embedding vector_cosine_ops)"
    )

    op.create_index(
        "idx_document_chunks_parent_doc",
        "document_chunks",
        ["parent_doc_id"],
    )


def downgrade() -> None:
    """Downgrade schema."""

    op.drop_index(
        "idx_document_chunks_parent_doc",
        table_name="document_chunks",
    )

    op.execute(
        "DROP INDEX IF EXISTS idx_document_chunks_embedding"
    )

    op.execute(
        "DROP INDEX IF EXISTS idx_courses_embedding"
    )

    op.execute(
        "DROP INDEX IF EXISTS idx_concept_taxonomy_embedding"
    )

    op.drop_table("chunk_domain_tags")
    op.drop_table("document_chunks")
    op.drop_table("courses")
    op.drop_table("concept_taxonomy")