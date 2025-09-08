"""Database models for the TelegramGroupHelperBot."""

from datetime import datetime

from sqlalchemy import BigInteger, Column, DateTime, Integer, String, Text, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Message(Base):
    """Model for storing message data."""
    
    __tablename__ = "messages"
    
    # Define a unique constraint on chat_id and message_id
    __table_args__ = (
        UniqueConstraint('chat_id', 'message_id', name='uix_chat_message'),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    message_id = Column(BigInteger, index=True, nullable=False)
    chat_id = Column(BigInteger, index=True, nullable=False)
    user_id = Column(BigInteger, nullable=True)
    username = Column(String(255), nullable=True)
    text = Column(Text)
    language = Column(String(8))
    date = Column(DateTime, default=datetime.utcnow, index=True)
    reply_to_message_id = Column(BigInteger, nullable=True) 