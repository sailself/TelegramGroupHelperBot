"""Create messages table

Revision ID: 1a2b3c4d5e6f
Revises: 
Create Date: 2023-05-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '1a2b3c4d5e6f'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'messages',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('chat_id', sa.BigInteger(), nullable=False),
        sa.Column('user_id', sa.BigInteger(), nullable=False),
        sa.Column('username', sa.String(255), nullable=True),
        sa.Column('text', sa.Text(), nullable=True),
        sa.Column('language', sa.String(8), nullable=True),
        sa.Column('date', sa.DateTime(), nullable=False),
        sa.Column('reply_to_message_id', sa.BigInteger(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_messages_chat_id'), 'messages', ['chat_id'], unique=False)
    op.create_index(op.f('ix_messages_date'), 'messages', ['date'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_messages_date'), table_name='messages')
    op.drop_index(op.f('ix_messages_chat_id'), table_name='messages')
    op.drop_table('messages') 