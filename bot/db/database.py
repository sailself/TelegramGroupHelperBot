"""Database connection and session management."""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, List, Optional, Tuple
from datetime import datetime

from sqlalchemy import select as sa_select, update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.exc import IntegrityError

from bot.config import DATABASE_URL
from bot.db.models import Base, Message

# Set up logging
logger = logging.getLogger(__name__)

# Create async engine and session maker
engine = create_async_engine(DATABASE_URL)
async_session = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

# Message queue system
message_queue = asyncio.Queue()

async def init_db() -> None:
    """Initialize the database and create tables if they don't exist."""
    try:
        # Drop tables first if they exist to ensure schema is up to date
        # Comment out the next line in production to avoid losing data
        # await drop_all_tables()
        
        # Create all tables based on the models
        async with engine.begin() as conn:
            # Create tables based on the models
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("Database tables created successfully")
        
        # Start the database writer task
        asyncio.create_task(db_writer())
        logger.info("Database writer task started")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise


async def drop_all_tables() -> None:
    """Drop all tables in the database.
    
    WARNING: This will delete all data. Use with caution!
    """
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        logger.info("All tables dropped successfully")
    except Exception as e:
        logger.error(f"Error dropping tables: {e}")
        raise


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Get a database session as an async context manager.
    
    Usage:
        async with get_session() as session:
            # Use session here
    
    Yields:
        AsyncSession: The database session.
    """
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_last_n_text_messages(
    chat_id: int, n: int, exclude_commands: bool = True
) -> List[Message]:
    """Get the last n text messages from a chat.
    
    Args:
        chat_id: The ID of the chat.
        n: The number of messages to retrieve.
        exclude_commands: Whether to exclude command messages.
        
    Returns:
        A list of Message objects.
    """
    async with get_session() as session:
        try:
            # Build the base query
            stmt = (
                sa_select(Message)
                .where(Message.chat_id == chat_id)
                .where(Message.text.isnot(None))
            )
            
            # Add command filtering if needed
            if exclude_commands:
                stmt = stmt.where(~Message.text.startswith("/"))
            
            # Order and limit
            stmt = stmt.order_by(Message.date.desc()).limit(n)
            
            # Execute query
            result = await session.execute(stmt)
            messages = result.scalars().all()
                
            # Return in chronological order
            return list(reversed(messages))
        except Exception as e:
            logger.error(f"Error retrieving messages: {e}")
            # Return empty list on error
            return []
        
async def get_messages_from_id(
    chat_id: int, from_message_id: int, exclude_commands: bool = True
) -> List[Message]:
    """Get the all the text messages from a chat begin with specific message id.
    
    Args:
        chat_id: The ID of the chat.
        from_message_id: The ID of the messages that begin with.
        exclude_commands: Whether to exclude command messages.
        
    Returns:
        A list of Message objects.
    """
    async with get_session() as session:
        try:
            # Build the base query
            stmt = (
                sa_select(Message)
                .where(Message.chat_id == chat_id)
                .where(Message.text.isnot(None))
                .where(Message.message_id >= from_message_id)
            )
            
            # Add command filtering if needed
            if exclude_commands:
                stmt = stmt.where(~Message.text.startswith("/"))
            
            # Order and limit
            stmt = stmt.order_by(Message.date.desc())
            
            # Execute query
            result = await session.execute(stmt)
            messages = result.scalars().all()
                
            # Return in chronological order
            return list(reversed(messages))
        except Exception as e:
            logger.error(f"Error retrieving messages: {e}")
            # Return empty list on error
            return []


async def db_writer() -> None:
    """Background task to write messages to the database from the queue with upsert support.
    
    Handles INSERT or UPDATE based on unique constraint of (chat_id, message_id).
    """
    while True:
        try:
            # Get next message data from the queue
            message_data = await message_queue.get()
            
            # Extract key identifiers
            chat_id = message_data.get("chat_id")
            message_id = message_data.get("message_id")
            
            async with get_session() as session:
                try:
                    # Check if a record with the same chat_id and message_id exists
                    stmt = sa_select(Message).where(
                        Message.chat_id == chat_id,
                        Message.message_id == message_id
                    )
                    result = await session.execute(stmt)
                    existing_message = result.scalars().first()
                    
                    if existing_message:
                        # Update existing record
                        logger.debug(f"Updating existing message: chat_id={chat_id}, message_id={message_id}")
                        for key, value in message_data.items():
                            setattr(existing_message, key, value)
                    else:
                        # Insert new record
                        logger.debug(f"Inserting new message: chat_id={chat_id}, message_id={message_id}")
                        message = Message(**message_data)
                        session.add(message)
                        
                    await session.commit()
                except IntegrityError as e:
                    # Handle race condition if a record was inserted between our check and update
                    logger.warning(f"IntegrityError during message upsert: {e}")
                    await session.rollback()
                    
                    # Try to update existing record
                    stmt = sa_update(Message).where(
                        Message.chat_id == chat_id,
                        Message.message_id == message_id
                    ).values(**message_data)
                    await session.execute(stmt)
                    await session.commit()
                except Exception as e:
                    logger.error(f"Error in db_writer: {e}")
                    await session.rollback()
                
            # Mark task as done
            message_queue.task_done()
            
        except Exception as e:
            logger.error(f"Error in db_writer: {e}")
            # Still mark as done to avoid blocking the queue
            message_queue.task_done()
            
        # Small sleep to avoid CPU hogging
        await asyncio.sleep(0.1)


async def queue_message_insert(
    user_id: int,
    username: str,
    text: str,
    language: str,
    date: datetime,
    reply_to_message_id: Optional[int] = None,
    chat_id: Optional[int] = None,
    message_id: Optional[int] = None
) -> None:
    """Queue a message for insertion into the database.
    
    Args:
        user_id: The ID of the user who sent the message.
        username: The username of the user who sent the message.
        text: The text of the message.
        language: The detected language of the message.
        date: The date the message was sent.
        reply_to_message_id: The ID of the message this message is replying to.
        chat_id: The ID of the chat where the message was sent.
        message_id: The ID of the message.
    """
    message_data = {
        "user_id": user_id,
        "username": username,
        "text": text,
        "language": language,
        "date": date,
        "reply_to_message_id": reply_to_message_id,
        "chat_id": chat_id or user_id,  # Default to user_id for private chats
        "message_id": message_id or 0,  # Default to 0 if not provided
    }
    
    await message_queue.put(message_data)


async def select_messages(chat_id: int, limit: int = 10) -> List[Message]:
    """Select messages from the database.
    
    Args:
        chat_id: The ID of the chat.
        limit: The maximum number of messages to return.
        
    Returns:
        A list of Message objects.
    """
    return await get_last_n_text_messages(chat_id, limit) 

async def select_messages_from_id(chat_id: int, message_id: int) -> List[Message]:
    """Select messages from the database.
    
    Args:
        chat_id: The ID of the chat.
        message_id: The message id to begin with.
        
    Returns:
        A list of Message objects.
    """
    return await get_messages_from_id(chat_id, message_id)