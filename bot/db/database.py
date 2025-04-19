"""Database connection and session management."""

import asyncio
from typing import AsyncGenerator, Dict, List, Optional, Tuple

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.future import select

from bot.config import DATABASE_URL
from bot.db.models import Base, Message

# Create async engine and session maker
engine = create_async_engine(DATABASE_URL)
async_session = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

# Queue for message logging
message_queue: asyncio.Queue[Dict] = asyncio.Queue()


async def init_db() -> None:
    """Initialize the database and create tables if they don't exist."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Get a database session."""
    async with async_session() as session:
        yield session


async def db_writer() -> None:
    """Background task to write messages to the database in batches."""
    batch_size = 10
    batch: List[Dict] = []
    
    while True:
        try:
            # Get message from queue or wait for new one
            message_data = await message_queue.get()
            batch.append(message_data)
            
            # If we have a full batch or queue is empty, commit to database
            if len(batch) >= batch_size or message_queue.empty():
                if batch:
                    async with async_session() as session:
                        async with session.begin():
                            session.add_all([Message(**msg) for msg in batch])
                        await session.commit()
                    batch = []
            
            # If queue is empty, wait a short time before checking again
            if message_queue.empty():
                await asyncio.sleep(0.1)
                
        except Exception as e:
            print(f"Error in db_writer: {e}")
            await asyncio.sleep(1)  # Back off on error


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
    async with async_session() as session:
        query = select(Message).where(
            Message.chat_id == chat_id,
            Message.text.isnot(None)
        ).order_by(Message.date.desc()).limit(n)
        
        result = await session.execute(query)
        messages = result.scalars().all()
        
        if exclude_commands:
            messages = [msg for msg in messages if not msg.text.startswith("/")]
            
        # Return in chronological order
        return list(reversed(messages)) 