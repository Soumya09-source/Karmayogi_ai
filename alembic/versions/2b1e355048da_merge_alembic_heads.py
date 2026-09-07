"""merge alembic heads

Revision ID: 2b1e355048da
Revises: 549e8970bf90, b2af5d8aa207
Create Date: 2026-09-07 18:36:58.933805

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2b1e355048da'
down_revision: Union[str, Sequence[str], None] = ('549e8970bf90', 'b2af5d8aa207')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
