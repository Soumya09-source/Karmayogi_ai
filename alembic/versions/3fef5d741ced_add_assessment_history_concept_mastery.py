"""add assessment_history, concept_mastery

Revision ID: 3fef5d741ced
Revises: b6f4a0c9e2d1
Create Date: 2026-09-05 10:26:24.135882

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "3fef5d741ced"
down_revision: Union[str, Sequence[str], None] = "b6f4a0c9e2d1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""

    op.create_table(
        "concept_mastery",
        sa.Column("employee_id", sa.String(), nullable=False),
        sa.Column("concept_id", sa.String(), nullable=False),
        sa.Column("p_l0", sa.Float(), nullable=False),
        sa.Column("p_t", sa.Float(), nullable=False),
        sa.Column("p_g", sa.Float(), nullable=False),
        sa.Column("p_s", sa.Float(), nullable=False),
        sa.Column("p_mastery_current", sa.Float(), nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("last_updated", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ["employee_id"],
            ["employee_profile.employee_id"],
        ),
        sa.ForeignKeyConstraint(
            ["concept_id"],
            ["concept_taxonomy.canonical_concept_id"],
        ),
        sa.PrimaryKeyConstraint("employee_id", "concept_id"),
    )

    op.create_table(
        "assessment_history",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("employee_id", sa.String(), nullable=False),
        sa.Column("mcq_id", sa.String(), nullable=False),
        sa.Column("concept_id", sa.String(), nullable=False),
        sa.Column("correct", sa.Boolean(), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["employee_id"],
            ["employee_profile.employee_id"],
        ),
        sa.ForeignKeyConstraint(
            ["mcq_id"],
            ["mcqs.id"],
        ),
        sa.ForeignKeyConstraint(
            ["concept_id"],
            ["concept_taxonomy.canonical_concept_id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(
        "ix_assessment_history_session_id",
        "assessment_history",
        ["session_id"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""

    op.drop_index(
        "ix_assessment_history_session_id",
        table_name="assessment_history",
    )
    op.drop_table("assessment_history")
    op.drop_table("concept_mastery")