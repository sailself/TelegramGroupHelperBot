"""Message handlers for the TelegramGroupHelperBot."""

import asyncio
import time
from datetime import datetime
from typing import Dict, Final, Optional, cast

import langdetect
from telegram import Message as TelegramMessage
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from bot.config import FACTCHECK_PROMPT, RATE_LIMIT, TLDR_PROMPT
from bot.db import get_last_n_text_messages, message_queue
from bot.llm import call_gemini, stream_gemini

# Rate limiting
user_rate_limits: Dict[int, float] = {}


def is_rate_limited(user_id: int) -> bool:
    """Check if a user is rate limited.
    
    Args:
        user_id: The user ID to check.
        
    Returns:
        True if the user is rate limited, False otherwise.
    """
    current_time = time.time()
    last_request_time = user_rate_limits.get(user_id, 0)
    
    if current_time - last_request_time < RATE_LIMIT:
        return True
    
    user_rate_limits[user_id] = current_time
    return False


async def log_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log a message to the database.
    
    Args:
        update: The update object.
        context: The context object.
    """
    if not update.effective_message or not update.effective_message.text:
        return
    
    message = update.effective_message
    
    # Try to detect the language
    language = "unknown"
    try:
        if message.text:
            language = langdetect.detect(message.text)
    except langdetect.LangDetectException:
        pass
    
    # Queue message for database insertion
    await message_queue.put({
        "chat_id": message.chat_id,
        "user_id": message.from_user.id if message.from_user else 0,
        "username": message.from_user.username if message.from_user else "",
        "text": message.text,
        "language": language,
        "date": datetime.utcnow(),
        "reply_to_message_id": message.reply_to_message.message_id if message.reply_to_message else None,
    })


async def tldr_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /tldr command.
    
    Args:
        update: The update object.
        context: The context object.
    """
    if not update.effective_message or not context.args:
        await update.effective_message.reply_text("Please specify the number of messages to summarize, e.g. /tldr 10")
        return
    
    user_id = update.effective_message.from_user.id if update.effective_message.from_user else 0
    
    # Check rate limiting
    if is_rate_limited(user_id):
        await update.effective_message.reply_text("⏳ Please wait a moment…")
        return
    
    try:
        num_messages = int(context.args[0])
        if num_messages <= 0 or num_messages > 100:
            await update.effective_message.reply_text("Please provide a number between 1 and 100.")
            return
    except ValueError:
        await update.effective_message.reply_text("Please provide a valid number, e.g. /tldr 10")
        return
    
    # Get the messages from the database
    messages = await get_last_n_text_messages(
        update.effective_message.chat_id, 
        num_messages
    )
    
    if not messages:
        await update.effective_message.reply_text("No messages found to summarize.")
        return
    
    # Format the messages
    formatted_messages = "\n".join(
        f"{msg.username or 'User'}: {msg.text}" for msg in messages
    )
    
    # Call Gemini for summarization
    try:
        status_message = await update.effective_message.reply_text("正在总结中，请稍候...")
        
        response = await call_gemini(
            system_prompt=TLDR_PROMPT,
            user_content=formatted_messages,
            # No response_language because TLDR always replies in Chinese
        )
        
        # Truncate to 200 characters
        if len(response) > 200:
            response = response[:200] + "..."
        
        await status_message.edit_text(response)
        
    except Exception as e:
        print(f"Error in tldr_handler: {e}")
        await status_message.edit_text("⚠️ Gemini service temporarily unavailable")


async def factcheck_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /factcheck command.
    
    Args:
        update: The update object.
        context: The context object.
    """
    if not update.effective_message or not update.effective_message.reply_to_message:
        await update.effective_message.reply_text("❌ 请先回复一条消息再使用 /factcheck")
        return
    
    user_id = update.effective_message.from_user.id if update.effective_message.from_user else 0
    
    # Check rate limiting
    if is_rate_limited(user_id):
        await update.effective_message.reply_text("⏳ Please wait a moment…")
        return
    
    # Get the replied message text
    replied_message = update.effective_message.reply_to_message
    if not replied_message.text:
        await update.effective_message.reply_text("❌ Only text messages can be fact-checked.")
        return
    
    # Try to detect language
    language = "unknown"
    try:
        language = langdetect.detect(replied_message.text)
    except langdetect.LangDetectException:
        pass
    
    # Call Gemini for fact checking with streaming
    try:
        status_message = await update.effective_message.reply_text("Checking facts...")
        
        # Get streaming queue
        queue = await stream_gemini(
            system_prompt=FACTCHECK_PROMPT,
            user_content=replied_message.text,
            # No response_language needed as FACTCHECK_PROMPT tells to match the input language
        )
        
        # Stream responses by updating the message
        last_update_time = time.time()
        last_response = None
        
        while True:
            response = await queue.get()
            
            # End of stream
            if response is None:
                break
                
            current_time = time.time()
            # Update message every 0.5 seconds or on final response
            if current_time - last_update_time >= 0.5:
                try:
                    await status_message.edit_text(
                        response, 
                        parse_mode=ParseMode.MARKDOWN
                    )
                    last_update_time = current_time
                    last_response = response
                except Exception as e:
                    print(f"Error editing message: {e}")
        
        # Final update if needed
        if last_response != response:
            try:
                await status_message.edit_text(
                    response, 
                    parse_mode=ParseMode.MARKDOWN
                )
            except Exception as e:
                print(f"Error on final message edit: {e}")
                
    except Exception as e:
        print(f"Error in factcheck_handler: {e}")
        await status_message.edit_text("⚠️ Gemini service temporarily unavailable")


async def q_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /q command.
    
    Args:
        update: The update object.
        context: The context object.
    """
    if not update.effective_message:
        return
    
    user_id = update.effective_message.from_user.id if update.effective_message.from_user else 0
    
    # Check rate limiting
    if is_rate_limited(user_id):
        await update.effective_message.reply_text("⏳ Please wait a moment…")
        return
    
    # Extract query
    query_text = " ".join(context.args) if context.args else ""
    
    # If replying to a message, add that context
    if update.effective_message.reply_to_message and update.effective_message.reply_to_message.text:
        replied_text = update.effective_message.reply_to_message.text
        query_text = f"> {replied_text}\n\n{query_text}"
    
    if not query_text.strip():
        await update.effective_message.reply_text("Please provide a question, e.g. /q Who wrote Hamlet?")
        return
    
    # Detect language
    language = "English"  # Default language
    try:
        detected = langdetect.detect(query_text)
        if detected:
            language = detected
    except langdetect.LangDetectException:
        pass
    
    # Call Gemini for answering
    try:
        status_message = await update.effective_message.reply_text("Thinking...")
        
        response = await call_gemini(
            system_prompt="You are a helpful assistant.",
            user_content=query_text,
            response_language=language
        )
        
        await status_message.edit_text(response)
        
    except Exception as e:
        print(f"Error in q_handler: {e}")
        await status_message.edit_text("⚠️ Gemini service temporarily unavailable") 