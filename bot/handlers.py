"""Message handlers for the TelegramGroupHelperBot."""

import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional
import markdown
from bs4 import BeautifulSoup

import langid
import requests
from bs4 import BeautifulSoup
from html2text import html2text
from telegram import Update, InputMediaPhoto
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
from bot.db.database import queue_message_insert, select_messages_from_id
from bot.llm import call_gemini, stream_gemini, generate_image_with_gemini, download_media, generate_video_with_veo
from io import BytesIO

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
            tag_name = child.name
            node_content = {}

            # Handle table specifically
            if tag_name == 'table':
                table_html_str = str(child)
                # Remove newlines from table_html_str to prevent html2text from adding too many blank lines
                table_html_str = table_html_str.replace('\n', '')
                table_text = html2text.html2text(table_html_str)
                # Strip leading/trailing whitespace, but preserve internal formatting
                stripped_table_text = table_text.strip()
                if stripped_table_text: # Only add if there's content
                    nodes.append({'tag': 'pre', 'children': [stripped_table_text]})
                continue

            # Supported tags list (excluding table, handled above)
            supported_tags = [
                'p', 'aside', 'ul', 'ol', 'li', 'blockquote', 'pre', 'code',
                'a', 'b', 'strong', 'i', 'em', 'u', 's', 'br', 'hr',
                'img', 'video', 'figcaption', 'figure'
            ]
            header_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']

            if tag_name in header_tags:
                node_content['tag'] = 'h4' # Map all h1-h6 to h4
            elif tag_name in supported_tags:
                node_content['tag'] = tag_name
            else:
                # Unsupported tag (that is not a table): recursively process children
                nodes.extend(html_to_telegraph_nodes(child))
                continue # Skip creating a node for the unsupported tag itself

            # Add attributes if needed for supported tags
            if tag_name == 'a' and child.get('href'):
                node_content['attrs'] = {'href': child['href']}
            elif tag_name == 'img' and child.get('src'):
                node_content['attrs'] = {'src': child['src']}
                if child.get('alt'):
                    node_content['attrs']['alt'] = child['alt']
            # Note: video, figure, figcaption attributes might be needed too if they use them.
            
            # Process children recursively for the current supported tag
            processed_children = html_to_telegraph_nodes(child)
            if processed_children:
                node_content['children'] = processed_children
            
            nodes.append(node_content)
            
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
            # This is an error from the API, not a Python exception in this block.
            # If we wanted to log the response_data itself for debugging, that's different.
            # For now, not adding exc_info=True as 'e' is not in this scope.
            logger.error(f"Failed to create Telegraph page: {response_data.get('error')}")
            return None
            
    except Exception as e:
        logger.error(f"Error creating Telegraph page: {e}", exc_info=True)
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
                    logger.error(f"Failed to send response as plain text: {plain_e}", exc_info=True)
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
        # Default to 100 messages if no number is provided
        count = 100
        
        if context.args and len(context.args) > 0:
            try:
                count = int(context.args[0])
                
                # Ensure count is reasonable
                if count < 1:
                    count = 100
                elif count > 500:
                    count = 500
                    
            except ValueError:
                # If the argument is not a valid number, default to 100
                count = 100
        
        # Send a "processing" message
        processing_message = await update.effective_message.reply_text(
            "Summarizing recent messages..."
        )

        from_message_id = update.effective_message.message_id - count
        
        # Fetch recent messages
        # messages = await select_messages(chat_id=update.effective_chat.id, limit=count)
        messages = await select_messages_from_id(chat_id=update.effective_chat.id, message_id=from_message_id)
        
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
        prompt = TLDR_SYSTEM_PROMPT.format(bot_name=TELEGRAPH_AUTHOR_NAME)
        
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
        logger.error(f"Error in tldr_handler: {e}", exc_info=True)
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
    
    # Get the message to fact-check
    replied_message = update.effective_message.reply_to_message
    message_to_check = replied_message.text or replied_message.caption or ""
    
    image_data_list: List[bytes] = []
    video_data: Optional[bytes] = None
    video_mime_type: Optional[str] = None
    
    # Video processing (takes precedence)
    if replied_message.video:
        logger.info(f"Fact-checking video: {replied_message.video.file_id} in chat {replied_message.chat.id}")
        try:
            video_file = await context.bot.get_file(replied_message.video.file_id)
            video_mime_type = replied_message.video.mime_type
            dl_video_data = await download_media(video_file.file_path)
            if dl_video_data:
                video_data = dl_video_data
                image_data_list = [] # Clear images if video is present
                logger.info(f"Video {replied_message.video.file_id} downloaded for fact-check. MIME: {video_mime_type}")
                if not message_to_check: # If no caption, use default prompt for video
                    message_to_check = "Please fact-check this video."
            else:
                logger.error(f"Failed to download video {replied_message.video.file_id} for fact-check.")
        except Exception as e:
            logger.error(f"Error processing video for fact-check: {e}", exc_info=True)
    
    # Photo processing (only if video was not processed)
    if not video_data and replied_message.photo:
        # Simplified to handle only the single photo from the replied message
        photo_size = replied_message.photo[-1]
        try:
            file = await context.bot.get_file(photo_size.file_id)
            img_bytes = await download_media(file.file_path)
            if img_bytes:
                image_data_list.append(img_bytes)
                logger.info(f"Added single image to fact-check list from message {replied_message.message_id}.")
        except Exception as e:
            logger.error(f"Error downloading single image for fact-check: {e}", exc_info=True)

    if not message_to_check and not image_data_list and not video_data: # Corrected Python 'and'
        await update.effective_message.reply_text(
            "Cannot fact-check an empty message with no media (image/video)."
        )
        return
    
    # Default prompt if only media is present
    if not message_to_check:
        if video_data:
            message_to_check = "Please analyze this video and verify any claims or content shown in it."
        elif image_data_list:
            message_to_check = "Please analyze these images and verify any claims or content shown in them."
        
    language, _ = langid.classify(message_to_check)
    
    processing_message_text = "Fact-checking message..."
    if video_data:
        processing_message_text = "Analyzing video and fact-checking content..."
    elif image_data_list:
        processing_message_text = f"Analyzing {len(image_data_list)} image(s) and fact-checking content..."
    
    processing_message = await update.effective_message.reply_text(processing_message_text)
    
    try:
        # Format the system prompt with the current date
        current_datetime = datetime.utcnow().strftime("%H:%M:%S %B %d, %Y")
        system_prompt = FACTCHECK_SYSTEM_PROMPT.format(current_datetime=current_datetime)
        use_pro_model = bool(video_data or image_data_list) # Use Pro model if media is present
        
        # Get response from Gemini with fact checking
        response_queue = await stream_gemini(
            system_prompt=system_prompt,
            user_content=message_to_check,
            response_language=language,
            image_data_list=image_data_list if image_data_list else None,
            video_data=video_data,
            video_mime_type=video_mime_type,
            use_pro_model=True
        )
        
        # Process the streamed response
        full_response = ""

        resp_msg = "Checking facts"
        if video_data:
            resp_msg = "Analyzing video and checking facts"
        elif image_data_list:
            resp_msg = f"Analyzing {len(image_data_list)} image(s) and checking facts"
        
        if use_pro_model:
            resp_msg = f"{resp_msg} with Pro Model...\nIt could take longer than usual, please be patient."
        else:
            resp_msg = f"{resp_msg}..."
            
        await processing_message.edit_text(resp_msg, parse_mode=None)
        
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
        logger.error(f"Error in factcheck_handler: {e}", exc_info=True)
        try:
            await processing_message.edit_text(
                f"Error fact-checking message: {str(e)}"
            )
        except Exception: # Inner exception during error reporting
            pass

async def q_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /q command.

    Args:
        update: The update containing the message.
        context: The context object.
    """
    if update.effective_message is None or update.effective_chat is None:
        return

    if is_rate_limited(update.effective_message.from_user.id):
        await update.effective_message.reply_text(
            "Rate limit exceeded. Please try again later."
        )
        return

    try:
        query = " ".join(context.args) if context.args else ""
        image_data_list: List[bytes] = []
        video_data: Optional[bytes] = None
        video_mime_type: Optional[str] = None
        target_message_for_media = None
        media_group_id_to_log = None

        if update.effective_message.reply_to_message:
            target_message_for_media = update.effective_message.reply_to_message
            replied_text_content = target_message_for_media.text or target_message_for_media.caption or ""
            if replied_text_content:
                if query: 
                    query = f"Context from replied message: \"{replied_text_content}\"\n\nQuestion: {query}"
                else: 
                    query = replied_text_content
        else: 
            if update.effective_message.photo or update.effective_message.video: # Check current message for media
                 target_message_for_media = update.effective_message
                 if not query and update.effective_message.caption:
                     query = update.effective_message.caption

        if target_message_for_media:
            media_group_id_to_log = target_message_for_media.media_group_id
            # Video processing (takes precedence)
            if target_message_for_media.video:
                logger.info(f"Q handler processing video: {target_message_for_media.video.file_id}")
                try:
                    video_file = await context.bot.get_file(target_message_for_media.video.file_id)
                    dl_video_data = await download_media(video_file.file_path)
                    if dl_video_data:
                        video_data = dl_video_data
                        video_mime_type = target_message_for_media.video.mime_type
                        image_data_list = [] # Clear images
                        logger.info(f"Video {target_message_for_media.video.file_id} downloaded for /q. MIME: {video_mime_type}")
                    else:
                        logger.error(f"Failed to download video {target_message_for_media.video.file_id} for /q.")
                except Exception as e:
                    logger.error(f"Error processing video for /q: {e}", exc_info=True)
            
            # Photo processing (only if video was not processed)
            if not video_data and target_message_for_media.photo:
                # Simplified to handle only the single photo from the target message
                photo_size = target_message_for_media.photo[-1]
                try:
                    file = await context.bot.get_file(photo_size.file_id)
                    img_bytes = await download_media(file.file_path)
                    if img_bytes:
                        image_data_list.append(img_bytes)
                        logger.info(f"Added single image to /q list from message {target_message_for_media.message_id}.")
                except Exception as e:
                    logger.error(f"Error downloading single image for /q: {e}", exc_info=True)
        
        if not query and not image_data_list and not video_data: # Corrected Python 'and'
            await update.effective_message.reply_text(
                "Please provide a question, reply to media, or caption media with /q."
            )
            return

        if not query: # Default prompt if only media is present
            if video_data: query = "Please analyze this video."
            elif image_data_list: query = "Please analyze these image(s)."

        processing_message_text = "Processing your question..."
        if video_data:
            processing_message_text = "Analyzing video and processing your question..."
        elif image_data_list:
            processing_message_text = f"Analyzing {len(image_data_list)} image(s) and processing your question..."
        
        processing_message = await update.effective_message.reply_text(processing_message_text)
        
        language, _ = langid.classify(query)
        
        username = "Anonymous"
        if update.effective_sender: # Should be update.effective_message.from_user
            sender = update.effective_message.from_user
            if sender.full_name: username = sender.full_name
            elif sender.first_name and sender.last_name: username = f"{sender.first_name} {sender.last_name}"
            elif sender.first_name: username = sender.first_name
            elif sender.username: username = sender.username
        
        await queue_message_insert(
            user_id=update.effective_message.from_user.id, # Use from_user here
            username=username,
            text=f"Ask {TELEGRAPH_AUTHOR_NAME}: {query}", 
            language=language,
            date=update.effective_message.date,
            reply_to_message_id=update.effective_message.reply_to_message.message_id if update.effective_message.reply_to_message else None,
            chat_id=update.effective_chat.id,
            message_id=update.effective_message.message_id
        )
        
        current_datetime = datetime.utcnow().strftime("%H:%M:%S %B %d, %Y")
        system_prompt = Q_SYSTEM_PROMPT.format(current_datetime=current_datetime)
        use_pro_model = bool(video_data or image_data_list) # Use Pro model if media is present
        
        response = await call_gemini(
            system_prompt=system_prompt,
            user_content=query,
            image_data_list=image_data_list if image_data_list else None,
            video_data=video_data,
            video_mime_type=video_mime_type,
            use_pro_model=use_pro_model 
        )
        
        if response:
            await send_response(processing_message, response, "Answer to Your Question", ParseMode.MARKDOWN)
        else:
            await processing_message.edit_text(
                "I couldn't find an answer to your question. Please try rephrasing or asking something else."
            )
    
    except Exception as e:
        logger.error(f"Error in q_handler: {e}", exc_info=True)
        try:
            await update.effective_message.reply_text(
                f"Error processing your question: {str(e)}"
            )
        except Exception: # Inner exception during error reporting
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

        language, _ = langid.classify(message_text)
        
        username = "Anonymous"
        if update.effective_sender: # Should be update.effective_message.from_user
            sender = update.effective_message.from_user
            if sender.full_name: username = sender.full_name
            elif sender.first_name and sender.last_name: username = f"{sender.first_name} {sender.last_name}"
            elif sender.first_name: username = sender.first_name
            elif sender.username: username = sender.username

        await queue_message_insert(
            user_id=update.effective_message.from_user.id, # Use from_user here
            username=username,
            text=message_text, 
            language=language,
            date=update.effective_message.date,
            reply_to_message_id=update.effective_message.reply_to_message.message_id if update.effective_message.reply_to_message else None,
            chat_id=update.effective_chat.id,
            message_id=update.effective_message.message_id
        )

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
                    # await context.bot.send_photo(
                    #     chat_id=update.effective_chat.id,
                    #     photo=image_io,
                    #     caption=f"Image generated, hope you like it."
                    # )
                    await processing_message.edit_media(media=InputMediaPhoto(media=image_io, caption=f"Image generated, hope you like it."))
                    # await processing_message.edit_text("Image generated, hope you like it.")
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

async def vid_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for the /vid command.
    
    This command uses Veo to generate a video based on the provided text and/or image.
    
    Args:
        update: The update containing the message.
        context: The context object.
    """
    if update.effective_user is None or update.effective_message is None:
        return

    user_id = update.effective_user.id
    
    # Rate Limiting
    if is_rate_limited(user_id):
        await update.effective_message.reply_text(
            "You're sending commands too quickly. Please wait a moment before trying again."
        )
        return

    logger.info(f"Received /vid command from user {user_id}")

    # Prompt Extraction
    message_text = update.effective_message.text or ""
    prompt = ""
    if message_text.startswith("/vid"):
        prompt = message_text[len("/vid"):].strip()

    # Image Handling
    image_data_bytes: Optional[bytes] = None
    replied_message = update.effective_message.reply_to_message

    if replied_message:
        if replied_message.photo:
            logger.info(f"Replying to photo for /vid command.")
            photo = replied_message.photo[-1]  # Get the largest photo
            try:
                photo_file = await context.bot.get_file(photo.file_id)
                image_data_bytes = await download_media(photo_file.file_path)
                if image_data_bytes:
                    logger.info(f"Image downloaded successfully for /vid command.")
                else:
                    logger.warning("Failed to download image for /vid.")
            except Exception as e:
                logger.error(f"Error downloading image for /vid: {e}", exc_info=True)
                await update.effective_message.reply_text("Error downloading image. Please try again.")
                return
        
        # Use caption as prompt if no command prompt and caption exists
        if not prompt and replied_message.caption:
            prompt = replied_message.caption.strip()
            logger.info(f"Using caption from replied message as prompt: '{prompt}'")


    # Default Prompt or Usage Message
    if not prompt and not image_data_bytes:
        await update.effective_message.reply_text(
            "Please provide a prompt for the video, or reply to an image with a prompt (or use image caption as prompt).\n"
            "Usage: /vid [text prompt]\n"
            "Or reply to an image: /vid [optional text prompt]"
        )
        return
    elif not prompt and image_data_bytes:
        prompt = "Generate a video from this image."
        logger.info(f"Using default prompt for image-only /vid: '{prompt}'")

    processing_message = await update.effective_message.reply_text(
        "Processing video request... This may take a few minutes."
    )

    try:
        language, _ = langid.classify(message_text)
        
        username = "Anonymous"
        if update.effective_sender: # Should be update.effective_message.from_user
            sender = update.effective_message.from_user
            if sender.full_name: username = sender.full_name
            elif sender.first_name and sender.last_name: username = f"{sender.first_name} {sender.last_name}"
            elif sender.first_name: username = sender.first_name
            elif sender.username: username = sender.username

        await queue_message_insert(
            user_id=update.effective_message.from_user.id, # Use from_user here
            username=username,
            text=message_text, 
            language=language,
            date=update.effective_message.date,
            reply_to_message_id=update.effective_message.reply_to_message.message_id if update.effective_message.reply_to_message else None,
            chat_id=update.effective_chat.id,
            message_id=update.effective_message.message_id
        )

        # Call Video Generation
        logger.info(f"Calling generate_video_with_veo with prompt: '{prompt}' and image_data: {'present' if image_data_bytes else 'absent'}")
        video_bytes, video_mime_type = await generate_video_with_veo(
            system_prompt="You are a helpful video generation assistant.",
            user_prompt=prompt,
            image_data=image_data_bytes
        )

        # Response Handling
        if video_bytes:
            logger.info(f"Video generated successfully. MIME type: {video_mime_type}, Size: {len(video_bytes)} bytes")
            video_file = BytesIO(video_bytes)
            video_file.name = 'generated_video.mp4' # Suggested name

            try:
                await update.effective_message.reply_video(
                    video=video_file,
                    caption="Here's your generated video!",
                    read_timeout=120, # Increased timeouts
                    write_timeout=120,
                    connect_timeout=60,
                    pool_timeout=60 
                )
                await processing_message.delete()
                logger.info("Video sent successfully and processing message deleted.")
            except Exception as e_telegram:
                logger.error(f"Error sending video via Telegram: {e_telegram}", exc_info=True)
                await processing_message.edit_text(
                    "Sorry, I generated the video but couldn't send it via Telegram. It might be too large or in an unsupported format."
                )
        else:
            logger.warning("Video generation failed (video_bytes is None).")
            await processing_message.edit_text(
                "Sorry, I couldn't generate the video. Please try a different prompt or image. The model might have limitations or be unavailable."
            )

    except Exception as e:
        logger.error(f"Error in vid_handler: {e}", exc_info=True)
        error_message_user = "Sorry, an unexpected error occurred while generating your video. Please try again later."
        if "timeout" in str(e).lower():
            error_message_user = "The video generation timed out. Please try a shorter prompt or a smaller image."
        elif "unsupported" in str(e).lower():
             error_message_user = "The video generation failed due to an unsupported format or feature. Please check your input."
        try:
            await processing_message.edit_text(error_message_user)
        except Exception: # If editing fails, just log
            logger.error("Failed to edit processing message with error.", exc_info=True)


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
        "• /img [description] - Generate or edit an image using Gemini\n"
        "• /vid [prompt] - Generate a video based on text and/or a replied-to image.\n"
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

    /vid - Generate a video
    Usage: `/vid [text prompt]` (optionally reply to an image)
    Or: `/vid` (replying to an image with an optional text prompt in the caption or command)

    /help - Show this help message
    """

    await update.effective_message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN) 