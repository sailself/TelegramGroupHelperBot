"""Message handlers for the TelegramGroupHelperBot."""

import json
import logging
import re
import time
import uuid
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, cast

import langid
import requests
from bs4 import BeautifulSoup
from html2text import html2text
from telegram import Update
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from bot.config import (
    RATE_LIMIT_SECONDS, 
    TELEGRAM_MAX_LENGTH, 
    TLDR_SYSTEM_PROMPT, 
    FACTCHECK_SYSTEM_PROMPT, 
    Q_SYSTEM_PROMPT,
    TELEGRAPH_ACCESS_TOKEN
)
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

async def create_telegraph_page(title: str, content: str) -> Optional[str]:
    """Create a Telegraph page with the provided content.
    
    Args:
        title: The title of the page.
        content: The content of the page as plain text.
        
    Returns:
        The URL of the created page, or None if creation failed.
    """
    try:
        # Convert content to Telegraph's node-based format
        # For simple text content, we'll split it into paragraphs
        paragraphs = content.split('\n\n')
        
        # Create a node structure for each paragraph
        nodes = []
        for paragraph in paragraphs:
            if paragraph.strip():  # Skip empty paragraphs
                # Create a paragraph node with text content
                nodes.append({
                    "tag": "p", 
                    "children": [paragraph.strip()]
                })
        
        # Create the page
        response = requests.post(
            'https://api.telegra.ph/createPage',
            data={
                'access_token': TELEGRAPH_ACCESS_TOKEN,
                'title': title,
                'content': json.dumps(nodes),  # This is the key change - properly formatted JSON
                'return_content': 'false'
            },
            timeout=10
        )
        
        response_data = response.json()
        
        if response_data.get('ok'):
            return response_data['result']['url']
        else:
            logger.error(f"Failed to create Telegraph page: {response_data.get('error')}")
            return None
            
    except Exception as e:
        logger.error(f"Error creating Telegraph page: {e}")
        return None

async def send_response(message, response, title="Response", parse_mode=ParseMode.MARKDOWN_V2):
    """Send a response, creating a Telegraph page if it's too long.
    
    Args:
        message: The message to edit with the response.
        response: The response text to send.
        title: The title for the Telegraph page if created.
        parse_mode: The parse mode to use for Telegram messages.
        
    Returns:
        None
    """
    # Proactively check if the message is too long
    if len(response) > TELEGRAM_MAX_LENGTH:
        logger.info(f"Response length {len(response)} exceeds threshold {TELEGRAM_MAX_LENGTH}, creating Telegraph page")
        telegraph_url = await create_telegraph_page(title, response)
        if telegraph_url:
            await message.edit_text(
                f"I have too much to say. Please view it here: {telegraph_url}"
            )
        else:
            # Fallback: try to send as plain text
            try:
                await message.edit_text(response)
            except BadRequest as e:
                # If still too long, truncate
                await message.edit_text(
                    f"{response[:TELEGRAM_MAX_LENGTH - 100]}...\n\n(Response was truncated due to length)"
                )
    else:
        # Message is within limits, try to send with formatting
        try:
            await message.edit_text(
                response,
                parse_mode=parse_mode
            )
        except Exception as e:
            logger.warning(f"Failed to send response with formatting: {e}")
            # If formatting fails, try to send as plain text
            try:
                await message.edit_text(response)
            except BadRequest as plain_e:
                if "Message_too_long" in str(plain_e):
                    # Handle the case where the message is unexpectedly too long
                    telegraph_url = await create_telegraph_page(title, response)
                    if telegraph_url:
                        await message.edit_text(
                            f"The response is too long for Telegram. View it here: {telegraph_url}"
                        )
                    else:
                        # Last resort: truncate
                        await message.edit_text(
                            f"{response[:TELEGRAM_MAX_LENGTH - 100]}...\n\n(Response was truncated due to length)"
                        )
                else:
                    logger.error(f"Failed to send response as plain text: {plain_e}")
                    await message.edit_text(
                        "Error: Failed to format response. Please try again."
                    )

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
    
    # Get the proper user display name
    if message.from_user:
        if message.from_user.full_name:
            username = message.from_user.full_name
        elif message.from_user.first_name and message.from_user.last_name:
            username = f"{message.from_user.first_name} {message.from_user.last_name}"
        elif message.from_user.first_name:
            username = message.from_user.first_name
        elif message.from_user.username:
            username = message.from_user.username
        else:
            username = "Anonymous"
    else:
        username = "Anonymous"
    
    # Queue message for insertion
    await queue_message_insert(
        user_id=message.from_user.id if message.from_user else 0,
        username=username,
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
                elif count > 200:
                    count = 200
                    
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
            # Use the username if available, otherwise use "Anonymous" as fallback
            username = msg.username if msg.username and msg.username.strip() else "Anonymous"
            if msg.reply_to_message_id:
                formatted_messages += f"msg[{msg.message_id}] reply_to[{msg.reply_to_message_id}] {timestamp} - {username}: {msg.text}\n\n"
            else:
                formatted_messages += f"msg[{msg.message_id}] {timestamp} - {username}: {msg.text}\n\n"
        
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
            await send_response(processing_message, response, "Message Summary", ParseMode.MARKDOWN_V2)
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
    
    # Check for images in the message
    image_url = None
    if update.effective_message.reply_to_message.photo:
        # Get the largest photo (last in the list)
        photo = update.effective_message.reply_to_message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        image_url = file.file_path
        logger.info(f"Found image in message to fact-check: {image_url}")
        
        # If there's a caption, use it as the message text
        if update.effective_message.reply_to_message.caption:
            message_to_check = update.effective_message.reply_to_message.caption
        # If there's no text at all, add a default prompt for image analysis
        elif not message_to_check:
            message_to_check = "Please analyze this image and verify any claims or content shown in it."
    
    if not message_to_check and not image_url:
        await update.effective_message.reply_text(
            "Cannot fact-check an empty message with no image."
        )
        return
        
    # Detect the language of the message
    language, _ = langid.classify(message_to_check)
    
    # Send a "processing" message
    processing_message = await update.effective_message.reply_text(
        "Fact-checking message..." if not image_url else "Analyzing image and fact-checking content..."
    )
    
    try:
        # Format the system prompt with the current date
        current_datetime = datetime.utcnow().strftime("%H:%M:%S %B %d, %Y")
        system_prompt = FACTCHECK_SYSTEM_PROMPT.format(current_datetime=current_datetime)
        
        # Get response from Gemini with fact checking, using the detected language
        response_queue = await stream_gemini(
            system_prompt=system_prompt,
            user_content=message_to_check,
            response_language=language,  # Pass the detected language
            image_url=image_url  # Pass the image URL if present
        )
        
        # Process the streamed response
        full_response = ""
        
        # Initialize the message
        await processing_message.edit_text(
            "Checking facts..." if not image_url else "Analyzing image and checking facts...", 
            parse_mode=None
        )
        
        # Process chunks from the queue
        while True:
            chunk = await response_queue.get()
            
            if chunk is None:
                # End of stream
                break
                
            full_response = chunk
        
        # Final update with complete response
        if full_response:
            await send_response(processing_message, full_response, "Fact Check Results", ParseMode.MARKDOWN_V2)
        else:
            await processing_message.edit_text(
                "Failed to fact-check the message. Please try again later."
            )
    
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
        
        # Check if there's an image in the message being replied to
        image_url = None
        
        # Check if the command is a reply to a message
        if update.effective_message.reply_to_message:
            # Check for text
            if not query:
                query = update.effective_message.reply_to_message.text or update.effective_message.reply_to_message.caption or ""
            
            # Check for images
            if update.effective_message.reply_to_message.photo:
                # Get the largest photo (last in the list)
                photo = update.effective_message.reply_to_message.photo[-1]
                file = await context.bot.get_file(photo.file_id)
                image_url = file.file_path
                logger.info(f"Found image in message to analyze: {image_url}")
                
                # If there's no text at all, add a default prompt for image analysis
                if not query:
                    query = "Please analyze this image and tell me what you see."
        
        if not query and not image_url:
            await update.effective_message.reply_text(
                "Please provide a question after /q or reply to a message with /q"
            )
            return
        
        # Send a "processing" message
        processing_message = await update.effective_message.reply_text(
            "Processing your question..." if not image_url else "Analyzing image and processing your question..."
        )
        
        # Detect language
        language, _ = langid.classify(query)
        
        if update.effective_sender:
            if update.effective_sender.full_name:
                username = update.effective_sender.full_name
            elif update.effective_sender.first_name and update.effective_sender.last_name:
                username = f"{update.effective_sender.first_name} {update.effective_sender.last_name}"
            elif update.effective_sender.first_name:
                username = update.effective_sender.first_name
            elif update.effective_sender.username:
                username = update.effective_sender.username
            else:
                username = "Anonymous"
        else:
            username = "Anonymous"
        # Log the query
        await queue_message_insert(
            user_id=update.effective_sender.id,
            username=username,
            text=query,
            language=language,
            date=update.effective_message.date,
            reply_to_message_id=update.effective_message.reply_to_message.message_id if update.effective_message.reply_to_message else None,
            chat_id=update.effective_chat.id,
            message_id=update.effective_message.message_id
        )
        
        # Format the system prompt with the current date
        current_datetime = datetime.utcnow().strftime("%H:%M:%S %B %d, %Y")
        system_prompt = Q_SYSTEM_PROMPT.format(current_datetime=current_datetime)
        
        # Get response from Gemini
        response = await call_gemini(
            system_prompt=system_prompt,
            user_content=query,
            image_url=image_url  # Pass the image URL if present
        )
        
        if response:
            await send_response(processing_message, response, "Answer to Your Question", ParseMode.MARKDOWN_V2)
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
        "• /factcheck - Reply to a message or image to fact-check it\n"
        "• /q [question] - Ask me any question or analyze images\n"
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
        "• /factcheck - Reply to a message or image to fact-check it.\n"
        "  Example: Reply to any message or photo with `/factcheck` to verify its claims or analyze visual content.\n\n"
        "• /q [question] - Ask me any question or analyze images.\n"
        "  Example: `/q What is the capital of France?` or reply to an image with `/q What's in this picture?`\n\n"
        "I'm powered by Google Gemini AI to provide accurate and helpful answers, including advanced image understanding capabilities."
    )
    
    await update.effective_message.reply_text(help_message) 