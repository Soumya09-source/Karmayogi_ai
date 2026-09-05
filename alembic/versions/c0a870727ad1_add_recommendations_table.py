"""add recommendations table

Revision ID: c0a870727ad1
Revises: f37abc50d2da
Create Date: 2026-09-05 10:48:18.643272

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c0a870727ad1'
down_revision: Union[str, Sequence[str], None] = '3fef5d741ced'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'recommendations',
        sa.Column('employee_id', sa.String(), nullable=False),
        sa.Column('gap_concept_id', sa.String(), nullable=False),
        sa.Column('recommended_course_id', sa.String(), nullable=False),
        sa.Column('similarity_score', sa.Float(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('status', sa.String(), nullable=False),
        sa.PrimaryKeyConstraint(
            'employee_id',
            'gap_concept_id',
            'recommended_course_id'
        )
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('recommendations')