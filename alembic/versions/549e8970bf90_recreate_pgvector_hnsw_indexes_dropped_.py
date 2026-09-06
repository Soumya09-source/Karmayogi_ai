"""recreate pgvector HNSW indexes dropped by 071653280ffc

Revision ID: 549e8970bf90
Revises: 8c673f94ac17
Create Date: 2026-09-05 20:05:33.794629

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '549e8970bf90'
down_revision: Union[str, Sequence[str], None] = '8c673f94ac17'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        'idx_concept_taxonomy_embedding', 'concept_taxonomy', ['embedding'],
        unique=False, postgresql_using='hnsw',
        postgresql_ops={'embedding': 'vector_cosine_ops'},
    )
    op.create_index(
        'idx_courses_embedding', 'courses', ['embedding'],
        unique=False, postgresql_using='hnsw',
        postgresql_ops={'embedding': 'vector_cosine_ops'},
    )
    op.create_index(
        'idx_document_chunks_embedding', 'document_chunks', ['embedding'],
        unique=False, postgresql_using='hnsw',
        postgresql_ops={'embedding': 'vector_cosine_ops'},
    )
    op.create_index(
        'idx_document_chunks_parent_doc', 'document_chunks', ['parent_doc_id'],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index('idx_document_chunks_parent_doc', table_name='document_chunks')
    op.drop_index('idx_document_chunks_embedding', table_name='document_chunks')
    op.drop_index('idx_courses_embedding', table_name='courses')
    op.drop_index('idx_concept_taxonomy_embedding', table_name='concept_taxonomy')
