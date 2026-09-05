"""add employee profile, mcqs, competency framework matrix

Revision ID: 75b5e64aa407
Revises: f37abc50d2da
Create Date: 2026-08-30
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "75b5e64aa407"
down_revision: Union[str, None] = "f37abc50d2da"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""

    op.create_table(
        "competency_framework_matrix",
        sa.Column("designation", sa.String(), nullable=False),
        sa.Column("competency_domain", sa.String(), nullable=False),
        sa.Column("expected_proficiency_level", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint(
            "designation",
            "competency_domain",
        ),
    )

    op.create_table(
        "employee_profile",
        sa.Column("employee_id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("designation", sa.String(), nullable=False),
        sa.Column("years_of_service", sa.Integer(), nullable=True),
        sa.Column("department", sa.String(), nullable=True),
        sa.PrimaryKeyConstraint("employee_id"),
    )

    op.create_table(
        "mcqs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("concept_id", sa.String(), nullable=False),
        sa.Column(
            "options",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column("correct_option_id", sa.String(), nullable=False),
        sa.Column("difficulty", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("times_served", sa.Integer(), nullable=False),
        sa.Column("times_correct", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["concept_id"],
            ["concept_taxonomy.canonical_concept_id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    """Downgrade schema."""

    op.drop_table("mcqs")
    op.drop_table("employee_profile")
    op.drop_table("competency_framework_matrix")