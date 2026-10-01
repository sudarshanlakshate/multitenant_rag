"""add chunk.page column for citation support

Revision ID: a1b2c3d4e5f6
Revises: 61df9ecad71a
Create Date: 2026-09-30 23:50:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = '61df9ecad71a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        'chunks',
        sa.Column('page', sa.Integer(), nullable=True),
    )
    op.create_index(
        op.f('ix_chunks_page'),
        'chunks',
        ['page'],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_chunks_page'), table_name='chunks')
    op.drop_column('chunks', 'page')