"""add question column to mcqs

Revision ID: 1a84ff812c83
Revises: 5b8bab21fd46
Create Date: 2026-09-08
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "1a84ff812c83"
down_revision: Union[str, Sequence[str], None] = "5b8bab21fd46"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "mcqs",
        sa.Column("question", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("mcqs", "question")