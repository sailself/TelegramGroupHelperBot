"""Database models for the TelegramGroupHelperBot."""

from datetime import datetime
from typing import Optional

from sqlalchemy import BigInteger, Column, DateTime, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Message(Base):
    """Model for storing message data."""
    
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    chat_id = Column(BigInteger, index=True)
    user_id = Column(BigInteger)
    username = Column(String(255))
    text = Column(Text)
    language = Column(String(8))
    date = Column(DateTime, default=datetime.utcnow, index=True)
    reply_to_message_id = Column(BigInteger, nullable=True) 