"""add documents table

Revision ID: 5b8bab21fd46
Revises: 2b1e355048da
Create Date: 2026-09-08 17:27:21.558291

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '5b8bab21fd46'
down_revision: Union[str, Sequence[str], None] = '2b1e355048da'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "documents",
        sa.Column("document_id", sa.String(), nullable=False),
        sa.Column("filename", sa.String(), nullable=False),
        sa.Column("uploaded_by", sa.String(), nullable=False),
        sa.Column(
            "status",
            sa.String(),
            nullable=False,
            server_default="sanity_pending",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("approved_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("document_id"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("documents")