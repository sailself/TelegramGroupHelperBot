"""Message handlers for the TelegramGroupHelperBot."""

import json
import logging
import re
import time
import uuid
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, cast
import markdown
from bs4 import BeautifulSoup

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
    TELEGRAPH_ACCESS_TOKEN,
    TELEGRAPH_AUTHOR_NAME,
    TELEGRAPH_AUTHOR_URL
)
from bot.db.database import select_messages, queue_message_insert
from bot.llm import call_gemini, stream_gemini, generate_image_with_gemini

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

def markdown_to_telegraph_nodes(md_content: str) -> List[Dict]:
    """Convert markdown content to Telegraph node format.
    
    Args:
        md_content: Content in markdown format
        
    Returns:
        List of Telegraph node objects
    """
    # Convert markdown to HTML
    html_content = markdown.markdown(md_content)
    
    # Parse HTML
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Convert to Telegraph nodes
    return html_to_telegraph_nodes(soup)

def html_to_telegraph_nodes(element) -> List[Dict]:
    """Convert HTML elements to Telegraph node format recursively.
    
    Args:
        element: BeautifulSoup element
        
    Returns:
        List of Telegraph node objects
    """
    nodes = []
    
    # Process all child elements
    for child in element.children:
        # Text node
        if child.name is None:
            text = child.string.strip()
            if text:
                nodes.append(text)
        # Element node
        else:
            node = {}
            
            # Map tag
            if child.name in ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'aside', 
                            'ul', 'ol', 'li', 'blockquote', 'pre', 'code', 
                            'a', 'b', 'strong', 'i', 'em', 'u', 's', 'br', 'hr', 
                            'img', 'video', 'figcaption', 'figure']:
                node['tag'] = child.name
            else:
                # Default to 'p' for unsupported tags
                node['tag'] = 'p'
            
            # Add attributes if needed
            if child.name == 'a' and child.get('href'):
                node['attrs'] = {'href': child['href']}
            elif child.name == 'img' and child.get('src'):
                node['attrs'] = {'src': child['src']}
                if child.get('alt'):
                    node['attrs']['alt'] = child['alt']
            
            # Process children recursively
            children = html_to_telegraph_nodes(child)
            if children:
                node['children'] = children
            
            nodes.append(node)
    
    return nodes

async def create_telegraph_page(title: str, content: str) -> Optional[str]:
    """Create a Telegraph page with the provided markdown content.
    
    Args:
        title: The title of the page.
        content: The content of the page in markdown format.
        
    Returns:
        The URL of the created page, or None if creation failed.
    """
    try:
        # Convert markdown to Telegraph nodes
        nodes = markdown_to_telegraph_nodes(content)
        
        # Create the page
        response = requests.post(
            'https://api.telegra.ph/createPage',
            data={
                'access_token': TELEGRAPH_ACCESS_TOKEN,
                'author_name': TELEGRAPH_AUTHOR_NAME,
                'author_url': TELEGRAPH_AUTHOR_URL,
                'title': title,
                'content': json.dumps(nodes),
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


async def send_response(message, response, title="Response", parse_mode=ParseMode.MARKDOWN):
    """Send a response, creating a Telegraph page if it's too long.
    
    Args:
        message: The message to edit with the response.
        response: The response text to send.
        title: The title for the Telegraph page if created.
        parse_mode: The parse mode to use for Telegram messages.
        
    Returns:
        None
    """
   # Count the number of lines in the response
    line_count = response.count('\n') + 1
    
    # Check if message exceeds line count threshold or character limit
    if line_count > 22 or len(response) > TELEGRAM_MAX_LENGTH:
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
                elif count > 500:
                    count = 500
                    
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
            await send_response(processing_message, response, "Message Summary", ParseMode.MARKDOWN)
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
        use_pro_model=False
        # Get response from Gemini with fact checking, using the detected language
        response_queue = await stream_gemini(
            system_prompt=system_prompt,
            user_content=message_to_check,
            response_language=language,  # Pass the detected language
            image_url=image_url,
            use_pro_model=use_pro_model  # Pass the image URL if present
        )
        
        # Process the streamed response
        full_response = ""

        resp_msg = "Checking facts" if not image_url else "Analyzing image and checking facts"
        if use_pro_model:
            resp_msg = f"{resp_msg} with Pro Model...\nIt could take longer than usual, please be patient."
        else:
            resp_msg = f"{resp_msg}..."
        # Initialize the message
        await processing_message.edit_text(
            resp_msg, 
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
            await send_response(processing_message, full_response, "Fact Check Results", ParseMode.MARKDOWN)
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
            reply = update.effective_message.reply_to_message.text or update.effective_message.reply_to_message.caption or ""
            if reply:
                if query:
                    query = f"Quote: {reply}\nQuestion: {query}"
                else:
                    query = reply
            
            # Check for images, the reply to message photo has higher priority
            photo = None
            if update.effective_message.photo:
                # Get the largest photo (last in the list)
                photo = update.effective_message.photo[-1]

            if update.effective_message.reply_to_message.photo:
                # Get the largest photo (last in the list)
                photo = update.effective_message.reply_to_message.photo[-1]

            if photo:
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
            await send_response(processing_message, response, "Answer to Your Question", ParseMode.MARKDOWN)
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

async def img_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for the /img command.
    
    This command uses Gemini to generate or edit an image based on the provided text.
    It can accept text or text with a reply to an image.
    
    Args:
        update: The update containing the message.
        context: The context object.
    """
    if update.effective_user is None or update.effective_message is None:
        return

    user_id = update.effective_user.id
    
    # Check rate limiting
    if is_rate_limited(user_id):
        await update.effective_message.reply_text(
            "You're sending commands too quickly. Please wait a moment before trying again."
        )
        return

    # Get message text without the command
    message_text = update.effective_message.text or ""
    if message_text.startswith("/img"):
        prompt = message_text[4:].strip()
    else:
        prompt = ""

    if not prompt:
        await update.effective_message.reply_text(
            "Please provide a description of the image you want to generate or edit. "
            "For example: /img a cat playing piano"
        )
        return

    # Get potential image from a replied message
    replied_message = update.effective_message.reply_to_message
    image_url = None
    
    if replied_message and replied_message.photo:
        # Get the largest photo (last in the array)
        photo = replied_message.photo[-1]
        photo_file = await context.bot.get_file(photo.file_id)
        image_url = photo_file.file_path

    # Send a processing message
    processing_message = await update.effective_message.reply_text(
        "Processing your image request... This may take a moment."
    )
    
    try:
        # Keep system prompt simple
        system_prompt = "Generate an image based on the description."
        
        # Log if this is a generation or editing request
        if image_url:
            logger.info(f"Processing image edit request: '{prompt}'")
        else:
            logger.info(f"Processing image generation request: '{prompt}'")
        
        # Generate the image using Gemini
        image_data = await generate_image_with_gemini(
            system_prompt=system_prompt,
            prompt=prompt,
            input_image_url=image_url
        )
        
        if image_data:
            # Send the generated image
            try:
                # Convert bytes to BytesIO for Telegram
                from io import BytesIO
                image_io = BytesIO(image_data)
                image_io.name = 'generated_image.jpg'
                
                # Try to send the image
                try:
                    await context.bot.send_photo(
                        chat_id=update.effective_chat.id,
                        photo=image_io,
                        caption=f"Generated image based on: {prompt[:200]}..." if len(prompt) > 200 else f"Generated image based on: {prompt}"
                    )
                    await processing_message.delete()
                except Exception as send_error:
                    logger.error(f"Error sending image via Telegram: {send_error}", exc_info=True)
                    
                    # Save the image to disk as a fallback for debugging
                    try:
                        import os
                        from datetime import datetime
                        
                        # Create a logs/images directory if it doesn't exist
                        os.makedirs("logs/images", exist_ok=True)
                        
                        # Generate a unique filename based on timestamp
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        file_path = f"logs/images/image_{timestamp}.jpg"
                        
                        # Save the image
                        with open(file_path, "wb") as f:
                            f.write(image_data)
                        
                        logger.info(f"Saved problematic image to {file_path}")
                        
                        # Notify the user
                        await processing_message.edit_text(
                            "I generated an image but couldn't send it through Telegram. "
                            "The image has been saved for debugging. Please try a different prompt."
                        )
                    except Exception as save_error:
                        logger.error(f"Error saving image to disk: {save_error}", exc_info=True)
                        await processing_message.edit_text(
                            "Sorry, I generated an image but couldn't send it through Telegram. The image format might be incompatible."
                        )
            except Exception as e:
                logger.error(f"Error sending image: {e}", exc_info=True)
                await processing_message.edit_text(
                    "Sorry, I generated an image but couldn't send it through Telegram. The image format might be incompatible."
                )
        else:
            # Different messages for generation vs editing
            if image_url:
                await processing_message.edit_text(
                    "I couldn't edit the image according to your request. The Gemini model may have limitations "
                    "with image editing capabilities. Please try:\n"
                    "1. Using a simpler edit description\n"
                    "2. Providing more specific details\n"
                    "3. Try a different type of edit or try again later"
                )
            else:
                await processing_message.edit_text(
                    "I couldn't generate an image based on your request. The Gemini model may have limitations with "
                    "image generation capabilities. Please try:\n"
                    "1. Using a simpler prompt\n"
                    "2. Providing more specific details\n"
                    "3. Try again later as model capabilities continue to improve"
                )
    except Exception as e:
        logger.error(f"Error in img_handler: {e}", exc_info=True)
        error_message = str(e).lower()
        
        if "not supported" in error_message or "unavailable" in error_message or "feature" in error_message:
            await processing_message.edit_text(
                "Sorry, image generation is not currently supported by the Gemini API. "
                "This feature may be available in the future."
            )
        elif "image_process_failed" in error_message:
            await processing_message.edit_text(
                "Sorry, there was an issue processing the generated image. "
                "Please try again with a different prompt or wait for a while before trying again."
            )
        else:
            await processing_message.edit_text(
                "Sorry, an error occurred while processing your image request. Please try again later."
            )

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
    
    help_text = """
*TelegramGroupHelperBot Commands*

/tldr - Summarize previous messages in the chat
Usage: Reply to a message with `/tldr` to summarize all messages between that message and the present.

/factcheck - Fact-check a statement or text
Usage: `/factcheck [statement]` or reply to a message with `/factcheck`

/q - Ask a question
Usage: `/q [your question]`

/img - Generate or edit an image using Gemini
Usage: `/img [description]` for generating a new image
Or reply to an image with `/img [description]` to edit that image

/help - Show this help message
"""

    await update.effective_message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN) 