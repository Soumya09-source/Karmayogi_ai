"""recreate pgvector HNSW indexes dropped by 071653280ffc

Revision ID: 8c673f94ac17
Revises: 071653280ffc
Create Date: 2026-09-05 20:03:48.812696

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8c673f94ac17'
down_revision: Union[str, Sequence[str], None] = '071653280ffc'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
