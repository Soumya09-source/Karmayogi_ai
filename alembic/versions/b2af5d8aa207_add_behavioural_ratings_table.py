"""add behavioural ratings table

Revision ID: b2af5d8aa207
Revises: c0a870727ad1
Create Date: 2026-09-06 10:48:30.413265

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "b2af5d8aa207"
down_revision: Union[str, Sequence[str], None] = "c0a870727ad1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create behavioural ratings table."""

    op.create_table(
        "behavioural_ratings",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("employee_id", sa.String(), nullable=False),
        sa.Column("rater_type", sa.String(), nullable=False),
        sa.Column("rater_id", sa.String(), nullable=False),
        sa.Column("competency_area", sa.String(), nullable=False),
        sa.Column("rating", sa.Integer(), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["employee_id"],
            ["employee_profile.employee_id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(
        "ix_behavioural_ratings_employee_id",
        "behavioural_ratings",
        ["employee_id"],
    )


def downgrade() -> None:
    """Drop behavioural ratings table."""

    op.drop_index(
        "ix_behavioural_ratings_employee_id",
        table_name="behavioural_ratings",
    )
    op.drop_table("behavioural_ratings")