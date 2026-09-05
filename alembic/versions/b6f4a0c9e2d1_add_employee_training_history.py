"""add employee training history table

Revision ID: b6f4a0c9e2d1
Revises: 9d3e7a21c8b4
Create Date: 2026-09-05 12:30:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "b6f4a0c9e2d1"
down_revision: Union[str, Sequence[str], None] = "75b5e64aa407"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""

    op.create_table(
        "employee_training_history",
        sa.Column("employee_id", sa.String(), nullable=False),
        sa.Column("course_id", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(
            ["employee_id"],
            ["employee_profile.employee_id"],
        ),
        sa.ForeignKeyConstraint(
            ["course_id"],
            ["courses.course_id"],
        ),
        sa.PrimaryKeyConstraint("employee_id", "course_id"),
    )


def downgrade() -> None:
    """Downgrade schema."""

    op.drop_table("employee_training_history")
