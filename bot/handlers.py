"""Message handlers for the TelegramGroupHelperBot."""

import logging
import re
import time
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, cast

import langid
from bs4 import BeautifulSoup
from html2text import html2text
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from bot.config import RATE_LIMIT_SECONDS, TLDR_SYSTEM_PROMPT, FACTCHECK_SYSTEM_PROMPT, Q_SYSTEM_PROMPT
from bot.db.database import select_messages, queue_message_insert
from bot.llm import call_gemini, stream_gemini

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Dictionary to store user rate limits
user_rate_limits: Dict[int, float] = {}

def is_rate_limited(user_id: int) -> bool:
    """Check if a user is rate limited.

    Args:
        user_id: The ID of the user to check.

    Returns:
        True if the user is rate limited, False otherwise.
    """
    global user_rate_limits
    current_time = time.time()
    
    if user_id in user_rate_limits:
        last_request_time = user_rate_limits[user_id]
        if current_time - last_request_time < RATE_LIMIT_SECONDS:
            return True
    
    user_rate_limits[user_id] = current_time
    return False

async def log_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log a message to the database.

    Args:
        update: The update containing the message.
        context: The context object.
    """
    if update.effective_message is None:
        return

    message = update.effective_message
    
    # Skip messages with no text content
    if not message.text and not message.caption:
        return
    
    text = message.text or message.caption or ""
    
    # Detect language
    language, _ = langid.classify(text)
    
    # Queue message for insertion
    await queue_message_insert(
        user_id=message.from_user.id if message.from_user else 0,
        username=message.from_user.username if message.from_user else "",
        text=text,
        language=language,
        date=message.date,
        reply_to_message_id=message.reply_to_message.message_id if message.reply_to_message else None,
        chat_id=update.effective_chat.id if update.effective_chat else None,
        message_id=message.message_id
    )

async def tldr_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /tldr command.

    Args:
        update: The update containing the message.
        context: The context object.
    """
    if update.effective_message is None or update.effective_chat is None:
        return

    # Check if user is rate limited
    if is_rate_limited(update.effective_message.from_user.id):
        await update.effective_message.reply_text(
            "Rate limit exceeded. Please try again later."
        )
        return

    try:
        # Default to 10 messages if no number is provided
        count = 10
        
        if context.args and len(context.args) > 0:
            try:
                count = int(context.args[0])
                
                # Ensure count is reasonable
                if count < 1:
                    count = 10
                elif count > 50:
                    count = 50
                    
            except ValueError:
                # If the argument is not a valid number, default to 10
                count = 10
        
        # Send a "processing" message
        processing_message = await update.effective_message.reply_text(
            "Summarizing recent messages..."
        )
        
        # Fetch recent messages
        messages = await select_messages(chat_id=update.effective_chat.id, limit=count)
        
        if not messages or len(messages) == 0:
            await processing_message.edit_text("No messages found to summarize.")
            return
        
        # Format messages for the LLM
        formatted_messages = "Recent conversation:\n\n"
        for msg in messages:
            timestamp = msg.date.strftime("%Y-%m-%d %H:%M:%S")
            username = msg.username or "Anonymous"
            formatted_messages += f"{timestamp} - {username}: {msg.text}\n\n"
        
        # Use the configured system prompt
        prompt = TLDR_SYSTEM_PROMPT
        
        # Detect language from the messages
        all_text = " ".join([msg.text for msg in messages])
        detected_language, _ = langid.classify(all_text)
        
        # Generate summary using Gemini
        response = await call_gemini(
            system_prompt=prompt,
            user_content=formatted_messages
        )
        
        if response:
            await processing_message.edit_text(
                response,
                parse_mode=ParseMode.MARKDOWN
            )
        else:
            await processing_message.edit_text(
                "Failed to generate a summary. Please try again later."
            )
            
    except Exception as e:
        logger.error(f"Error in tldr_handler: {e}")
        try:
            await update.effective_message.reply_text(
                f"Error summarizing messages: {str(e)}"
            )
        except Exception:
            pass

async def factcheck_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /factcheck command.

    Args:
        update: The update containing the message.
        context: The context object.
    """
    if update.effective_message is None or update.effective_chat is None:
        return

    # Check if user is rate limited
    if is_rate_limited(update.effective_message.from_user.id):
        await update.effective_message.reply_text(
            "Rate limit exceeded. Please try again later."
        )
        return

    # Check if the command is a reply to a message
    if not update.effective_message.reply_to_message:
        await update.effective_message.reply_text(
            "Please reply to a message to fact-check."
        )
        return

    # Get the message to fact-check
    message_to_check = update.effective_message.reply_to_message.text or update.effective_message.reply_to_message.caption or ""
    
    if not message_to_check:
        await update.effective_message.reply_text(
            "Cannot fact-check an empty message."
        )
        return
    
    # Detect language
    language, _ = langid.classify(message_to_check)
    
    # Log the message being fact-checked
    if update.effective_message.reply_to_message.from_user:
        await queue_message_insert(
            user_id=update.effective_message.reply_to_message.from_user.id,
            username=update.effective_message.reply_to_message.from_user.username or "",
            text=message_to_check,
            language=language,
            date=update.effective_message.reply_to_message.date,
            reply_to_message_id=None,
            chat_id=update.effective_chat.id,
            message_id=update.effective_message.reply_to_message.message_id
        )
    
    # Send a "processing" message
    processing_message = await update.effective_message.reply_text(
        "Fact-checking message..."
    )
    
    try:
        # Use streaming for fact-checking to update the user in real-time
        response_queue = await stream_gemini(
            system_prompt=FACTCHECK_SYSTEM_PROMPT,
            user_content=message_to_check
        )
        
        # Process chunks from the queue
        while True:
            chunk = await response_queue.get()
            if chunk is None:  # End of stream
                break
                
            try:
                await processing_message.edit_text(
                    chunk,
                    parse_mode=ParseMode.MARKDOWN
                )
            except Exception as e:
                # If we can't update with markdown, try without it
                try:
                    await processing_message.edit_text(chunk)
                except Exception as inner_e:
                    logger.error(f"Error updating fact-check response: {inner_e}")
                    # If we still can't update, stop streaming
                    break
    except Exception as e:
        logger.error(f"Error in factcheck_handler: {e}")
        try:
            await processing_message.edit_text(
                f"Error fact-checking message: {str(e)}"
            )
        except Exception:
            pass

async def q_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /q command.

    Args:
        update: The update containing the message.
        context: The context object.
    """
    if update.effective_message is None or update.effective_chat is None:
        return

    # Check if user is rate limited
    if is_rate_limited(update.effective_message.from_user.id):
        await update.effective_message.reply_text(
            "Rate limit exceeded. Please try again later."
        )
        return

    try:
        # Get the query from the message
        query = " ".join(context.args) if context.args else ""
        
        # Check if the command is a reply to a message
        if update.effective_message.reply_to_message and not query:
            query = update.effective_message.reply_to_message.text or update.effective_message.reply_to_message.caption or ""
        
        if not query:
            await update.effective_message.reply_text(
                "Please provide a question after /q or reply to a message with /q"
            )
            return
        
        # Send a "processing" message
        processing_message = await update.effective_message.reply_text(
            "Processing your question..."
        )
        
        # Detect language
        language, _ = langid.classify(query)
        
        # Log the query
        await queue_message_insert(
            user_id=update.effective_message.from_user.id,
            username=update.effective_message.from_user.username or "",
            text=query,
            language=language,
            date=update.effective_message.date,
            reply_to_message_id=update.effective_message.reply_to_message.message_id if update.effective_message.reply_to_message else None,
            chat_id=update.effective_chat.id,
            message_id=update.effective_message.message_id
        )
        
        # Get response from Gemini
        response = await call_gemini(
            system_prompt=Q_SYSTEM_PROMPT,
            user_content=query
        )
        
        if response:
            try:
                # Try to send with Markdown formatting
                await processing_message.edit_text(
                    response,
                    parse_mode=ParseMode.MARKDOWN
                )
            except Exception as e:
                # If Markdown fails, try to send as plain text
                logger.warning(f"Failed to send response with Markdown: {e}")
                try:
                    await processing_message.edit_text(response)
                except Exception as inner_e:
                    logger.error(f"Failed to send response as plain text: {inner_e}")
                    await processing_message.edit_text(
                        "Error: Failed to format response. Please try again."
                    )
        else:
            await processing_message.edit_text(
                "I couldn't find an answer to your question. Please try rephrasing or asking something else."
            )
    
    except Exception as e:
        logger.error(f"Error in q_handler: {e}")
        try:
            await update.effective_message.reply_text(
                f"Error processing your question: {str(e)}"
            )
        except Exception:
            pass

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /start command.

    Args:
        update: The update containing the message.
        context: The context object.
    """
    if update.effective_message is None or update.effective_chat is None:
        return
    
    welcome_message = (
        "👋 Hello! I'm TelegramGroupHelperBot, your AI assistant for this chat.\n\n"
        "I can help with the following commands:\n"
        "• /tldr [number] - Summarize recent messages (default: 10)\n"
        "• /factcheck - Reply to a message to fact-check it\n"
        "• /q [question] - Ask me any question\n"
        "• /help - Show this help message\n\n"
        "Just type one of these commands to get started!"
    )
    
    await update.effective_message.reply_text(welcome_message)


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /help command.

    Args:
        update: The update containing the message.
        context: The context object.
    """
    if update.effective_message is None or update.effective_chat is None:
        return
    
    help_message = (
        "Here's how you can use me:\n\n"
        "• /tldr [number] - Summarize recent messages. You can specify how many messages to include.\n"
        "  Example: `/tldr 20` will summarize the last 20 messages.\n\n"
        "• /factcheck - Reply to a message to fact-check it.\n"
        "  Example: Reply to any message with `/factcheck` to verify its claims.\n\n"
        "• /q [question] - Ask me any question.\n"
        "  Example: `/q What is the capital of France?`\n\n"
        "I'm powered by Google Gemini AI to provide accurate and helpful answers."
    )
    
    await update.effective_message.reply_text(help_message) 