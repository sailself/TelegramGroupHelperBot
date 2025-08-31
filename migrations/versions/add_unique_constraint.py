"""Add unique constraint to chat_id and message_id

Revision ID: add_unique_constraint
Revises: initial_migration
Create Date: 2023-09-18 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_unique_constraint'
down_revision = '53244dba3a71'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # SQLite doesn't support adding unique constraints directly
    # So we need to recreate the table with the constraint
    
    # First, create a new table with the constraint
    op.create_table(
        'messages_new',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('message_id', sa.BigInteger(), nullable=False),
        sa.Column('chat_id', sa.BigInteger(), nullable=False),
        sa.Column('user_id', sa.BigInteger(), nullable=True),
        sa.Column('username', sa.String(length=255), nullable=True),
        sa.Column('text', sa.Text(), nullable=True),
        sa.Column('language', sa.String(length=8), nullable=True),
        sa.Column('date', sa.DateTime(), nullable=True),
        sa.Column('reply_to_message_id', sa.BigInteger(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('chat_id', 'message_id', name='uix_chat_message')
    )
    
    # Insert only the rows with the largest id for each chat_id, message_id combination
    op.execute(
        """
        INSERT INTO messages_new (
            id, message_id, chat_id, user_id, username, text, language, date, reply_to_message_id
        )
        SELECT m.id, m.message_id, m.chat_id, m.user_id, m.username, m.text, m.language, m.date, m.reply_to_message_id
        FROM messages m
        INNER JOIN (
            SELECT chat_id, message_id, MAX(id) as max_id
            FROM messages
            GROUP BY chat_id, message_id
        ) subq ON m.chat_id = subq.chat_id AND m.message_id = subq.message_id AND m.id = subq.max_id
        """
    )
    
    # Drop the old table
    op.drop_table('messages')
    
    # Rename the new table to the original name
    op.rename_table('messages_new', 'messages')
    
    # Add indexes to maintain performance
    op.create_index(op.f('ix_messages_chat_id'), 'messages', ['chat_id'], unique=False)
    op.create_index(op.f('ix_messages_date'), 'messages', ['date'], unique=False)
    op.create_index(op.f('ix_messages_message_id'), 'messages', ['message_id'], unique=False)


def downgrade() -> None:
    # Remove the unique constraint by recreating the table without it
    op.create_table(
        'messages_old',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('message_id', sa.BigInteger(), nullable=True),
        sa.Column('chat_id', sa.BigInteger(), nullable=True),
        sa.Column('user_id', sa.BigInteger(), nullable=True),
        sa.Column('username', sa.String(length=255), nullable=True),
        sa.Column('text', sa.Text(), nullable=True),
        sa.Column('language', sa.String(length=8), nullable=True),
        sa.Column('date', sa.DateTime(), nullable=True),
        sa.Column('reply_to_message_id', sa.BigInteger(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Copy data
    op.execute(
        """
        INSERT INTO messages_old (
            id, message_id, chat_id, user_id, username, text, language, date, reply_to_message_id
        )
        SELECT id, message_id, chat_id, user_id, username, text, language, date, reply_to_message_id
        FROM messages
        """
    )
    
    # Drop the table with the constraint
    op.drop_table('messages')
    
    # Rename the old table back
    op.rename_table('messages_old', 'messages')
    
    # Recreate the indexes
    op.create_index(op.f('ix_messages_chat_id'), 'messages', ['chat_id'], unique=False)
    op.create_index(op.f('ix_messages_date'), 'messages', ['date'], unique=False)
    op.create_index(op.f('ix_messages_message_id'), 'messages', ['message_id'], unique=False)
