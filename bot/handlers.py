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
import re

from bot.config import (
    RATE_LIMIT_SECONDS, 
    TELEGRAM_MAX_LENGTH, 
    TLDR_SYSTEM_PROMPT, 
    FACTCHECK_SYSTEM_PROMPT, 
    Q_SYSTEM_PROMPT,
    PROFILEME_SYSTEM_PROMPT,
    PAINTME_SYSTEM_PROMPT,
    USER_HISTORY_MESSAGE_COUNT,
    TELEGRAPH_ACCESS_TOKEN,
    TELEGRAPH_AUTHOR_NAME,
    TELEGRAPH_AUTHOR_URL,
    USE_VERTEX_IMAGE,
    VERTEX_IMAGE_MODEL,
    PORTRAIT_SYSTEM_PROMPT,
)
from bot.db.database import ( # Reformatted for clarity
    queue_message_insert, 
    select_messages_from_id, 
    select_messages_by_user
)
from bot.llm import (
    call_gemini,
    generate_image_with_gemini, 
    generate_image_with_vertex,
    download_media, 
    generate_video_with_veo
)
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

def extract_youtube_urls(text: str, max_urls: int = 10):
    """Extract up to max_urls YouTube video URLs (including shorts) from the given text and replace them with a marker.
    Returns (modified_text, list_of_urls).
    """
    if not text:
        return text, []
    # Regex for YouTube URLs (full, short, and shorts)
    pattern = r"((?:https?://)?(?:www\.|m\.)?(?:youtube\.com/(?:watch\?v=|shorts/)|youtu\.be/)([\w-]{11})(?:[\?&][^\s]*)?)"
    matches = list(re.finditer(pattern, text))
    urls = []
    new_text = text
    count = 0
    for match in reversed(matches):  # reversed so replacement doesn't affect indices
        if count >= max_urls:
            break
        full_url = match.group(0)
        vid_id = match.group(2)
        url = f"https://www.youtube.com/watch?v={vid_id}"
        urls.insert(0, url)  # maintain order
        # Replace the entire matched URL in text
        start, end = match.span(0)
        new_text = new_text[:start] + f"YouTube_{vid_id}" + new_text[end:]
        count += 1
    return new_text, urls

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
    youtube_urls = None
    
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
    if message_to_check and not video_data and not image_data_list: # If message is present and no media is present, extract YouTube URLs
        # Extract YouTube URLs and replace in text
        message_to_check, youtube_urls = extract_youtube_urls(message_to_check)
    
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
    elif youtube_urls and len(youtube_urls) > 0:
        processing_message_text = f"Analyzing {len(youtube_urls)} YouTube video(s) and fact-checking content..."
    
    processing_message = await update.effective_message.reply_text(processing_message_text)
    
    try:
        # Format the system prompt with the current date
        current_datetime = datetime.utcnow().strftime("%H:%M:%S %B %d, %Y")
        system_prompt = FACTCHECK_SYSTEM_PROMPT.format(current_datetime=current_datetime)
        use_pro_model = bool(video_data or image_data_list) # Use Pro model if media is present
        
        # Get response from Gemini with fact checking
        full_response = await call_gemini(
            system_prompt=system_prompt,
            user_content=message_to_check,
            response_language=language,
            image_data_list=image_data_list if image_data_list else None,
            video_data=video_data,
            video_mime_type=video_mime_type,
            use_pro_model=use_pro_model,
            youtube_urls=youtube_urls,
            fact_check=True
        )

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
        youtube_urls = None

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
        
        if not query and not image_data_list and not video_data:
            await update.effective_message.reply_text(
                "Please provide a question, reply to media, or caption media with /q."
            )
            return

        if not query: # Default prompt if only media is present
            if video_data: query = "Please analyze this video."
            elif image_data_list: query = "Please analyze these image(s)."
        
        if query and not video_data and not image_data_list: # If query is present and no media is present, extract YouTube URLs
            # Extract YouTube URLs and replace in text
            query, youtube_urls = extract_youtube_urls(query)

        processing_message_text = "Processing your question..."
        if video_data:
            processing_message_text = "Analyzing video and processing your question..."
        elif image_data_list:
            processing_message_text = f"Analyzing {len(image_data_list)} image(s) and processing your question..."
        elif youtube_urls and len(youtube_urls) > 0:
            processing_message_text = f"Analyzing {len(youtube_urls)} YouTube video(s) and processing your question..."
        
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
            use_pro_model=use_pro_model,
            youtube_urls=youtube_urls
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

    # Get potential image from a replied message
    replied_message = update.effective_message.reply_to_message
    image_url = None
    
    if replied_message:
        if replied_message.photo:
            # Get the largest photo (last in the array)
            photo = replied_message.photo[-1]
            photo_file = await context.bot.get_file(photo.file_id)
            image_url = photo_file.file_path
        
        replied_text_content = replied_message.text or replied_message.caption or ""
        if replied_text_content:
            if prompt: 
                if not image_url:
                    prompt = f"{replied_text_content}\n\n{prompt}"
            else: 
                prompt = replied_text_content

    if not prompt:
        await update.effective_message.reply_text(
            "Please provide a description of the image you want to generate or edit. "
            "For example: /img a cat playing piano"
        )
        return


    # Send a processing message
    processing_message = await update.effective_message.reply_text(
        "Processing your image request... This may take a moment."
    )
    
    try:
        if USE_VERTEX_IMAGE and not image_url:
            logger.info(f"img_handler operating in Vertex AI mode for prompt: '{prompt}'")
            # For Vertex, we are not currently supporting editing an existing image URL.
            # The generate_image_with_vertex function currently only takes a prompt.
            if image_url:
                logger.warning("Vertex AI image generation called with an input image URL, but it's not currently supported by generate_image_with_vertex. Proceeding with prompt only.")
                # Optionally, notify user:
                # await processing_message.edit_text("Image editing is not currently supported with Vertex AI. Generating a new image from your prompt.")
            
            images_data_list = await generate_image_with_vertex(prompt=prompt)

            if images_data_list:
                if len(images_data_list) > 1:
                    logger.info(f"Vertex AI generated {len(images_data_list)} images. Sending as media group.")
                    media_group = []
                    for i, img_data in enumerate(images_data_list):
                        img_io = BytesIO(img_data)
                        img_io.name = f'vertex_image_{i}.jpg'
                        caption_text = f"Images generated with {VERTEX_IMAGE_MODEL}, hope you like them!" if i == 0 else None
                        media_group.append(InputMediaPhoto(media=img_io, caption=caption_text))
                    
                    # await processing_message.context.bot.send_media_group(chat_id=update.effective_chat.id, media=media_group)
                    await processing_message.delete()
                    await update.effective_message.reply_media_group(media=media_group)
                    logger.info("Media group sent and processing message deleted.")
                elif len(images_data_list) == 1:
                    logger.info("Vertex AI generated 1 image. Sending as single photo.")
                    image_data = images_data_list[0]
                    image_io = BytesIO(image_data)
                    image_io.name = 'vertex_image.jpg'
                    await processing_message.edit_media(media=InputMediaPhoto(media=image_io, caption=f"Image generated with {VERTEX_IMAGE_MODEL}, hope you like it."))
                    logger.info("Single image sent by editing processing message.")
                # else case (empty list) is handled by the next 'else'
            else:
                logger.error("Vertex AI image generation failed or returned empty list.")
                await processing_message.edit_text(f"Sorry, {VERTEX_IMAGE_MODEL} couldn't generate required images. Please change your prompt and try again.")

        else: # Fallback to Gemini
            logger.info(f"img_handler operating in Gemini mode for prompt: '{prompt}'")
            # Keep system prompt simple
            system_prompt = "Generate an image based on the description."
            
            # Log if this is a generation or editing request
            if image_url:
                logger.info(f"Gemini: Processing image edit request: '{prompt}'")
            else:
                logger.info(f"Gemini: Processing image generation request: '{prompt}'")

            image_data = await generate_image_with_gemini(
                system_prompt=system_prompt,
                prompt=prompt,
                input_image_url=image_url
            )

            if image_data:
                logger.info("Gemini: Image data received. Attempting to send.")
                try:
                    image_io = BytesIO(image_data)
                    image_io.name = 'gemini_generated_image.jpg'
                    await processing_message.edit_media(media=InputMediaPhoto(media=image_io, caption="Image generated with Gemini, hope you like it."))
                    logger.info("Gemini: Image sent successfully by editing processing message.")
                except Exception as send_error:
                    logger.error(f"Gemini: Error sending image via Telegram: {send_error}", exc_info=True)
                    # Fallback saving to disk logic (can be kept or removed if not essential for Vertex path)
                    try:
                        import os
                        from datetime import datetime
                        os.makedirs("logs/images", exist_ok=True)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        file_path = f"logs/images/gemini_image_{timestamp}.jpg"
                        with open(file_path, "wb") as f:
                            f.write(image_data)
                        logger.info(f"Gemini: Saved problematic image to {file_path}")
                        await processing_message.edit_text(
                            "I generated an image with Gemini but couldn't send it. Saved for debugging."
                        )
                    except Exception as save_error:
                        logger.error(f"Gemini: Error saving image to disk: {save_error}", exc_info=True)
                        await processing_message.edit_text(
                            "Sorry, I generated an image with Gemini but couldn't send it. Format might be incompatible."
                        )
            else:
                logger.warning("Gemini: Image generation failed (image_data is None).")
                # Different messages for generation vs editing
                if image_url:
                    await processing_message.edit_text(
                        "I couldn't edit the image with Gemini. Please try:\n"
                        "1. Simpler edit description\n2. More specific details\n3. Different edit type or try later"
                    )
                else:
                    await processing_message.edit_text(
                        "I couldn't generate an image with Gemini. Please try:\n"
                        "1. Simpler prompt\n2. More specific details\n3. Try again later"
                    )

        # Moved common message logging outside the conditional block for cleaner structure
        # This part is common whether Vertex or Gemini was used (or attempted)
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

    except Exception as e:
        # General error logging for the handler
        logger.error(f"Error in img_handler: {e}", exc_info=True)
        error_message_for_user = "Sorry, an unexpected error occurred while processing your image request."
        
        # More specific error messages based on context (e.g., if it's a known API issue)
        # This part can be refined if common errors from Vertex/Gemini need special user messages.
        # For example, if 'e' contains specific error codes or messages.
        # if "quota" in str(e).lower():
        #    error_message_for_user = "Image generation quota exceeded. Please try again later."
        # elif "unsafe" in str(e).lower(): # Example for content safety issues
        #    error_message_for_user = "The generated image might be unsafe. Please try a different prompt."
            
        try:
            await processing_message.edit_text(error_message_for_user)
        except Exception as edit_err: # Fallback if editing the message fails
            logger.error(f"Failed to edit processing message with error: {edit_err}", exc_info=True)
            # As a last resort, try sending a new message if the original processing_message is gone or uneditable
            # await update.effective_message.reply_text(error_message_for_user)


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
        "• /profileme - Generate your user profile based on your chat history.\n"
        "• /paintme - Generate an image representing you based on your chat history.\n"
        "• /help - Show this help message\n\n"
        "Just type one of these commands to get started!"
    )
    
    await update.effective_message.reply_text(welcome_message)


async def paintme_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /paintme command.

    Fetches user's chat history, generates an image prompt with Gemini,
    then generates an image and sends it to the user.
    """
    if update.effective_user is None or update.effective_message is None or update.effective_chat is None:
        return

    user_id = update.effective_user.id

    if is_rate_limited(user_id):
        await update.effective_message.reply_text("Rate limit exceeded. Please try again later.")
        return

    isPortrait = update.effective_message.text.lower().startswith("/portraitme")
    if isPortrait:
        system_prompt = PORTRAIT_SYSTEM_PROMPT
        reply_text_part = "create you a portrait"
    else:
        system_prompt = PAINTME_SYSTEM_PROMPT
        reply_text_part = "paint you a picture"

    processing_message = await update.effective_message.reply_text(
        f"Let me {reply_text_part} based on your recent chats... This might take a moment."
    )

    try:
        # 1. Fetch user's chat history using the new function and config
        user_messages = await select_messages_by_user(
            chat_id=update.effective_chat.id,
            user_id=user_id,
            limit=USER_HISTORY_MESSAGE_COUNT 
        )
        

        if not user_messages or len(user_messages) < 5: # Need a few messages at least
            await processing_message.edit_text(
                f"I don't have enough of your messages (at least 5 recent ones) in this chat to {reply_text_part}. Keep chatting!"
            )
            return

        # 2. Format chat history for Gemini prompt generation
        formatted_history = "User's recent messages in the group chat:\n\n"
        for msg in user_messages:
            formatted_history += f"- \"{msg.text}\"\n"
        
        # 3. Generate image prompt with Gemini
        image_prompt = await call_gemini(
            system_prompt=system_prompt, # Use imported constant
            user_content=formatted_history,
            use_search_grounding=False
        )

        if not image_prompt or "No response generated" in image_prompt:
            await processing_message.edit_text(
                "I couldn't come up with an image idea for you at this time. Please try again later."
            )
            return
        
        await processing_message.edit_text(
            f"Generated image prompt: \"{image_prompt}\". Now creating your masterpiece..."
        )

        # 4. Generate Image
        if USE_VERTEX_IMAGE:
            logger.info(f"{'portrait' if isPortrait else 'paint'}me_handler: Using Vertex AI for image generation with prompt: '{image_prompt}'")
            images_data_list = await generate_image_with_vertex(prompt=image_prompt, number_of_images=1) # Generate 1 for this command
            
            if images_data_list and images_data_list[0]:
                image_data = images_data_list[0]
                image_io = BytesIO(image_data)
                image_io.name = 'vertex_paintme_image.jpg'
                await update.effective_message.reply_photo(
                    photo=image_io, 
                    caption=f"Here's your artistic representation!\nPrompt: \"{image_prompt}\"\nModel: {VERTEX_IMAGE_MODEL}"
                )
                await processing_message.delete()
            else:
                logger.error(f"{'portrait' if isPortrait else 'paint'}me_handler: Vertex AI image generation failed or returned no image.")
                await processing_message.edit_text(f"Sorry, {VERTEX_IMAGE_MODEL} couldn't {reply_text_part}. Please try again.")
        
        else: # Use Gemini for image generation
            logger.info(f"{'portrait' if isPortrait else 'paint'}me_handler: Using Gemini for image generation with prompt: '{image_prompt}'")
            # System prompt for Gemini image generation can be simple
            gemini_image_system_prompt = "Generate an image based on the following description."
            image_data = await generate_image_with_gemini(
                system_prompt=gemini_image_system_prompt,
                prompt=image_prompt
            )

            if image_data:
                image_io = BytesIO(image_data)
                image_io.name = 'gemini_paintme_image.jpg'
                await update.effective_message.reply_photo(
                    photo=image_io, 
                    caption=f"Here's your artistic representation!\nPrompt: \"{image_prompt}\"\nModel: Gemini"
                )
                await processing_message.delete()
            else:
                logger.warning(f"{'portrait' if isPortrait else 'paint'}me_handler: Gemini image generation failed.")
                await processing_message.edit_text(f"Sorry, Gemini couldn't {reply_text_part}. Please try again.")
        
        # Log the command usage
        language, _ = langid.classify(formatted_history) # Use history for language context
        username = "Anonymous"
        if update.effective_message.from_user:
            sender = update.effective_message.from_user
            if sender.full_name: username = sender.full_name
            elif sender.first_name: username = sender.first_name
            elif sender.username: username = sender.username
        
        await queue_message_insert(
            user_id=user_id,
            username=username,
            text=update.effective_message.text, # Log the command itself
            language=language, # Could be 'xx' if history is very short/non-textual
            date=update.effective_message.date,
            chat_id=update.effective_chat.id,
            message_id=update.effective_message.message_id
        )

    except Exception as e:
        logger.error(f"Error in paintme_handler: {e}", exc_info=True)
        try:
            await processing_message.edit_text(
                f"Sorry, an unexpected error occurred while {'creating your portrait' if isPortrait else 'painting your picture'}: {str(e)}"
            )
        except Exception: 
            pass


async def profileme_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /profileme command.

    Args:
        update: The update containing the message.
        context: The context object.
    """
    if update.effective_user is None or update.effective_message is None or update.effective_chat is None:
        return

    user_id = update.effective_user.id

    # Check if user is rate limited
    if is_rate_limited(user_id):
        await update.effective_message.reply_text(
            "Rate limit exceeded. Please try again later."
        )
        return

    processing_message = await update.effective_message.reply_text(
        "Generating your profile... This might take a moment."
    )

    try:
        # Fetch user's chat history using the new function and config
        user_messages = await select_messages_by_user(
            chat_id=update.effective_chat.id,
            user_id=user_id,
            limit=USER_HISTORY_MESSAGE_COUNT
        )

        if not user_messages: # USER_HISTORY_MESSAGE_COUNT (e.g. 200) might be too high to always have messages.
            await processing_message.edit_text("I don't have enough of your messages in this chat to generate a profile yet (need some from your recent history). Keep chatting!")
            return

        # Format chat history for Gemini
        formatted_history = "Here is the user's recent chat history in this group:\n\n"
        for msg in user_messages:
            timestamp = msg.date.strftime("%Y-%m-%d %H:%M:%S")
            # We already know it's the user's message, so no need to repeat username
            formatted_history += f"{timestamp}: {msg.text}\n"
        
        system_prompt = PROFILEME_SYSTEM_PROMPT
        MAX_CUSTOM_PROMPT_LENGTH = 80
        custom_prompt = (update.effective_message.text or "").replace("/profileme", "", 1).strip()
        custom_prompt = custom_prompt[:MAX_CUSTOM_PROMPT_LENGTH]

        if custom_prompt:
            # Add style instruction if custom prompt is provided
            system_prompt = f"{system_prompt}\n\nStyle Instruction: {custom_prompt}"
        else:
            # Use a more specific default style
            system_prompt = f"{system_prompt}\n\nStyle Instruction: Keep the profile professional, friendly and respectful."

        # Generate profile using Gemini
        profile_response = await call_gemini(
            system_prompt=system_prompt,
            user_content=formatted_history,
            use_search_grounding=False # Probably not needed for profiling
        )

        if profile_response:
            await send_response(processing_message, profile_response, "Your User Profile")
        else:
            await processing_message.edit_text(
                "I couldn't generate a profile at this time. Please try again later."
            )

    except Exception as e:
        logger.error(f"Error in profileme_handler: {e}", exc_info=True)
        try:
            await processing_message.edit_text(
                f"Sorry, an error occurred while generating your profile: {str(e)}"
            )
        except Exception: # Inner exception during error reporting
            pass


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

    /profileme - Generate your user profile based on your chat history in this group.
    Usage: `/profileme` 
    Or: `/profileme [Language style of the profile]`

    /paintme - Generate an image representing you based on your chat history in this group.
    Usage: `/paintme`

    /portraitme - Generate a portrait of you based on your chat history in this group.
    Usage: `/portraitme`

    /help - Show this help message
    """

    await update.effective_message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN) 