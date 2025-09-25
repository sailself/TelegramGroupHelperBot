"""Question handling logic for TelegramGroupHelperBot."""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from functools import partial
from typing import Dict, List, Optional

import langid
from pycountry import languages
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from bot.config import (
    DEFAULT_Q_MODEL,
    DEEPSEEK_MODEL,
    ENABLE_OPENROUTER,
    GPT_MODEL,
    GROK_MODEL,
    LLAMA_MODEL,
    MODEL_SELECTION_TIMEOUT,
    OPENROUTER_API_KEY,
    Q_SYSTEM_PROMPT,
    QWEN_MODEL,
    TELEGRAPH_AUTHOR_NAME,
)
from bot.db.database import queue_message_insert
from bot.llm import call_gemini, call_openrouter, download_media

from .access import check_access_control, is_rate_limited
from .content import (
    download_telegraph_media,
    download_twitter_media,
    extract_telegraph_urls_and_content,
    extract_twitter_urls_and_content,
    extract_youtube_urls,
)
from .responses import send_response

logger = logging.getLogger(__name__)

MODEL_CALLBACK_PREFIX = "model_select:"
MODEL_GEMINI = "gemini"
MODEL_LLAMA = "llama"
MODEL_QWEN = "qwen"
MODEL_DEEPSEEK = "deepseek"
MODEL_GROK = "grok"

MODEL_CAPABILITIES = {
    MODEL_GEMINI: {"images": True, "video": True, "audio": True},
    MODEL_LLAMA: {"images": True, "video": True, "audio": True},
    MODEL_QWEN: {"images": False, "video": False, "audio": False},
    MODEL_DEEPSEEK: {"images": False, "video": False, "audio": False},
    MODEL_GROK: {"images": True, "video": False, "audio": False},
}

MODEL_CONFIG_VALUES = {
    MODEL_LLAMA: LLAMA_MODEL,
    MODEL_GROK: GROK_MODEL,
    MODEL_QWEN: QWEN_MODEL,
    MODEL_DEEPSEEK: DEEPSEEK_MODEL,
}

pending_q_requests: Dict[str, Dict] = {}
_cleanup_task: Optional[asyncio.Task] = None


def is_model_configured(model_key: str) -> bool:
    """Return True if the model has a configured identifier."""
    if model_key == MODEL_GEMINI:
        return True
    model_value = MODEL_CONFIG_VALUES.get(model_key)
    return bool(model_value and model_value.strip())


def is_openrouter_available() -> bool:
    """Check if OpenRouter is enabled and API key is available."""
    return ENABLE_OPENROUTER and OPENROUTER_API_KEY is not None and OPENROUTER_API_KEY.strip() != ""


def create_model_selection_keyboard(
    *,
    has_media: bool = False,
    has_images: bool = False,
    has_av_media: bool = False,
) -> InlineKeyboardMarkup:
    """Create inline keyboard for model selection.

    Args:
        has_media: If True, only show media-capable models
        has_images: Indicates the request includes image content
        has_av_media: Indicates the request includes video or audio content

    Returns:
        InlineKeyboardMarkup with model selection buttons
    """
    gemini_button = InlineKeyboardButton(
        "Gemini 2.5 ✨", callback_data=f"{MODEL_CALLBACK_PREFIX}{MODEL_GEMINI}"
    )
    llama_button = InlineKeyboardButton(
        "Llama 4", callback_data=f"{MODEL_CALLBACK_PREFIX}{MODEL_LLAMA}"
    )
    grok_button = InlineKeyboardButton(
        "Grok 4", callback_data=f"{MODEL_CALLBACK_PREFIX}{MODEL_GROK}"
    )
    qwen_button = InlineKeyboardButton(
        "Qwen 3", callback_data=f"{MODEL_CALLBACK_PREFIX}{MODEL_QWEN}"
    )
    deepseek_button = InlineKeyboardButton(
        "DeepSeek 3.1", callback_data=f"{MODEL_CALLBACK_PREFIX}{MODEL_DEEPSEEK}"
    )

    if has_media:
        keyboard: List[List[InlineKeyboardButton]] = [[gemini_button]]
        if is_model_configured(MODEL_LLAMA):
            keyboard[0].append(llama_button)

        if has_av_media:
            return InlineKeyboardMarkup(keyboard)

        if has_images and is_model_configured(MODEL_GROK):
            keyboard.append([grok_button])

        return InlineKeyboardMarkup(keyboard)

    keyboard: List[List[InlineKeyboardButton]] = [[gemini_button]]
    if is_model_configured(MODEL_LLAMA):
        keyboard[0].append(llama_button)

    second_row: List[InlineKeyboardButton] = []
    if is_model_configured(MODEL_GROK):
        second_row.append(grok_button)
    if is_model_configured(MODEL_QWEN):
        second_row.append(qwen_button)
    if second_row:
        keyboard.append(second_row)

    third_row: List[InlineKeyboardButton] = []
    if is_model_configured(MODEL_DEEPSEEK):
        third_row.append(deepseek_button)
    if third_row:
        keyboard.append(third_row)

    return InlineKeyboardMarkup(keyboard)


async def process_q_request_with_gemini(
    update: Update,
    processing_message,
    query: str,
    original_query: str,
    image_data_list: Optional[List[bytes]],
    video_data: Optional[bytes],
    video_mime_type: Optional[str],
    audio_data: Optional[bytes],
    audio_mime_type: Optional[str],
    youtube_urls: Optional[List[str]]
) -> None:
    """Process Q request directly with Gemini (original behavior when OpenRouter is disabled)."""
    try:
        # Get language
        language = languages.get(alpha_2=langid.classify(query)[0]).name

        # Get username
        username = "Anonymous"
        if update.effective_sender:
            sender = update.effective_message.from_user
            if sender.full_name:
                username = sender.full_name
            elif sender.first_name and sender.last_name:
                username = f"{sender.first_name} {sender.last_name}"
            elif sender.first_name:
                username = sender.first_name
            elif sender.username:
                username = sender.username

        # For database logging, use the original query
        db_query_text = original_query
        if update.effective_message.reply_to_message:
            replied_text_original = (
                update.effective_message.reply_to_message.text
                or update.effective_message.reply_to_message.caption
                or ""
            )
            if replied_text_original:
                if db_query_text:
                    db_query_text = f'Context from replied message: "{replied_text_original}"\n\nQuestion: {db_query_text}'
                else:
                    db_query_text = replied_text_original

        # Log database entry
        await queue_message_insert(
            user_id=update.effective_message.from_user.id,
            username=username,
            text=f"Ask {TELEGRAPH_AUTHOR_NAME}: {db_query_text}",
            language=language,
            date=update.effective_message.date,
            reply_to_message_id=(
                update.effective_message.reply_to_message.message_id
                if update.effective_message.reply_to_message
                else None
            ),
            chat_id=update.effective_chat.id,
            message_id=update.effective_message.message_id,
        )

        # Prepare system prompt
        current_datetime = datetime.utcnow().strftime("%H:%M:%S %B %d, %Y")
        system_prompt = Q_SYSTEM_PROMPT.format(
            current_datetime=current_datetime, language=language
        )
        
        # Determine if pro model should be used
        use_pro_model = bool(
            video_data or image_data_list or audio_data or youtube_urls
        )

        # Call Gemini
        response = await call_gemini(
            system_prompt=system_prompt,
            user_content=query,
            image_data_list=image_data_list if image_data_list else None,
            video_data=video_data,
            video_mime_type=video_mime_type,
            use_pro_model=use_pro_model,
            youtube_urls=youtube_urls,
            audio_data=audio_data,
            audio_mime_type=audio_mime_type,
        )

        if response:
            resp_text = response if isinstance(response, str) else response.get("final", "")
            await send_response(
                processing_message,
                resp_text,
                "Answer to Your Question",
                ParseMode.MARKDOWN,
            )
        else:
            await processing_message.edit_text(
                "I couldn't find an answer to your question. Please try rephrasing or asking something else.",
            )

    except Exception as e:
        logger.error("Error in process_q_request_with_gemini: %s", e, exc_info=True)
        try:
            await processing_message.edit_text(f"Error processing your question: {str(e)}")
        except Exception:
            pass


async def process_q_request_with_specific_model(
    update: Update,
    processing_message,
    query: str,
    original_query: str,
    image_data_list: Optional[List[bytes]],
    video_data: Optional[bytes],
    video_mime_type: Optional[str],
    audio_data: Optional[bytes],
    audio_mime_type: Optional[str],
    youtube_urls: Optional[List[str]],
    call_model,
    model_name: Optional[str]
) -> None:
    """Process Q request with a specific model."""
    try:
        # Get language
        language = languages.get(alpha_2=langid.classify(query)[0]).name

        # Get username
        username = "Anonymous"
        if update.effective_sender:
            sender = update.effective_message.from_user
            if sender.full_name:
                username = sender.full_name
            elif sender.first_name and sender.last_name:
                username = f"{sender.first_name} {sender.last_name}"
            elif sender.first_name:
                username = sender.first_name
            elif sender.username:
                username = sender.username

        # For database logging, use the original query
        db_query_text = original_query
        if update.effective_message.reply_to_message:
            replied_text_original = (
                update.effective_message.reply_to_message.text
                or update.effective_message.reply_to_message.caption
                or ""
            )
            if replied_text_original:
                if db_query_text:
                    db_query_text = f'Context from replied message: "{replied_text_original}"\n\nQuestion: {db_query_text}'
                else:
                    db_query_text = replied_text_original

        # Log database entry
        await queue_message_insert(
            user_id=update.effective_message.from_user.id,
            username=username,
            text=f"Ask {TELEGRAPH_AUTHOR_NAME}: {db_query_text}",
            language=language,
            date=update.effective_message.date,
            reply_to_message_id=(
                update.effective_message.reply_to_message.message_id
                if update.effective_message.reply_to_message
                else None
            ),
            chat_id=update.effective_chat.id,
            message_id=update.effective_message.message_id,
        )

        # Prepare system prompt
        current_datetime = datetime.utcnow().strftime("%H:%M:%S %B %d, %Y")
        system_prompt = Q_SYSTEM_PROMPT.format(
            current_datetime=current_datetime, language=language
        )
        
        # Determine if pro model should be used
        supports_video = model_name == LLAMA_MODEL or model_name is None
        supports_audio = supports_video
        supports_images = supports_video or model_name == GROK_MODEL

        temp_image_data_list = image_data_list if supports_images else None
        temp_video_data = video_data if supports_video else None
        temp_audio_data = audio_data if supports_audio else None
        temp_video_mime_type = video_mime_type if supports_video else None
        temp_audio_mime_type = audio_mime_type if supports_audio else None

        use_pro_model = bool(
            youtube_urls
            or temp_video_data
            or temp_image_data_list
            or temp_audio_data
        )

        # Call the specified model
        response = await call_model(
            system_prompt=system_prompt,
            user_content=query,
            image_data_list=temp_image_data_list if temp_image_data_list else None,
            video_data=temp_video_data,
            video_mime_type=temp_video_mime_type,
            use_pro_model=use_pro_model,
            youtube_urls=youtube_urls,
            audio_data=temp_audio_data,
            audio_mime_type=temp_audio_mime_type,
        )

        if response:
            if model_name == GPT_MODEL and isinstance(response, dict):
                analysis = response.get("analysis")
                final = response.get("final") or ""
                resp_text = ""
                if analysis:
                    resp_text += f"*Thought process*\n{analysis}\n\n"
                resp_text += f"*Final answer*\n{final}"
            else:
                resp_text = response if isinstance(response, str) else response.get("final", "")
            
            if model_name:
                resp_text = f"{resp_text}\n\n_Model: {model_name}_"
            
            await send_response(
                processing_message,
                resp_text,
                "Answer to Your Question",
                ParseMode.MARKDOWN,
            )
        else:
            await processing_message.edit_text(
                "I couldn't find an answer to your question. Please try rephrasing or asking something else.",
            )

    except Exception as e:
        logger.error("Error in process_q_request_with_specific_model: %s", e, exc_info=True)
        try:
            await processing_message.edit_text(f"Error processing your question: {str(e)}")
        except Exception:
            pass


async def q_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    call_model=None,
    model_name: Optional[str] = None,
) -> None:
    """Handle the /q command.

    Args:
        update: The update containing the message.
        context: The context object.
    """
    if update.effective_message is None or update.effective_chat is None:
        return

    # Check access control
    if not await check_access_control(update, "q"):
        return

    if is_rate_limited(update.effective_message.from_user.id):
        await update.effective_message.reply_text(
            "Rate limit exceeded. Please try again later."
        )
        return

    try:
        original_query = " ".join(context.args) if context.args else ""
        query = original_query  # Keep original for database logging
        message_entity_text = (
            update.effective_message.text
            or update.effective_message.caption
            or ""
        )

        image_data_list: List[bytes] = []
        video_data: Optional[bytes] = None
        video_mime_type: Optional[str] = None
        target_message_for_media = None
        youtube_urls = None
        language = None
        audio_data: Optional[bytes] = None
        audio_mime_type: Optional[str] = None
        telegraph_contents = []
        twitter_contents: List[dict] = []

        if update.effective_message.reply_to_message:
            target_message_for_media = update.effective_message.reply_to_message
            replied_text_content = (
                target_message_for_media.text or target_message_for_media.caption or ""
            )
            if replied_text_content:
                # Extract Telegraph content if present in replied message
                original_replied_text = replied_text_content
                replied_text_content, telegraph_contents = (
                    await extract_telegraph_urls_and_content(
                        replied_text_content, target_message_for_media.entities
                    )
                )
                replied_text_content, reply_twitter_contents = await extract_twitter_urls_and_content(
                    replied_text_content,
                    target_message_for_media.entities,
                    source_text=original_replied_text,
                )
                if reply_twitter_contents:
                    twitter_contents.extend(reply_twitter_contents)

                if query:
                    language = languages.get(
                        alpha_2=langid.classify(query)[0]
                    ).name  # Response language should follow question
                    query = f'Context from replied message: "{replied_text_content}"\n\nQuestion: {query}'
                else:
                    query = replied_text_content
        else:
            if (
                update.effective_message.photo
                or update.effective_message.video
                or update.effective_message.sticker
            ):  # Check current message for media
                target_message_for_media = update.effective_message
                if not query and update.effective_message.caption:
                    query = update.effective_message.caption

        if target_message_for_media:
            # Video processing (takes precedence)
            if target_message_for_media.video:
                logger.info(
                    "Q handler processing video: %s",
                    target_message_for_media.video.file_id,
                )
                try:
                    video_file = await context.bot.get_file(
                        target_message_for_media.video.file_id
                    )
                    dl_video_data = await download_media(video_file.file_path)
                    if dl_video_data:
                        video_data = dl_video_data
                        video_mime_type = target_message_for_media.video.mime_type
                        image_data_list = []  # Clear images
                        logger.info(
                            "Video %s downloaded for /q. MIME: %s",
                            target_message_for_media.video.file_id,
                            video_mime_type,
                        )
                    else:
                        logger.error(
                            "Failed to download video %s for /q.",
                            target_message_for_media.video.file_id,
                        )
                except Exception:  # noqa: BLE001
                    logger.error("Error processing video for /q", exc_info=True)

            # Audio processing (takes precedence)
            if not video_data:
                audio = (
                    target_message_for_media.audio
                    if target_message_for_media.audio
                    else target_message_for_media.voice
                )
                if audio:
                    logger.info("Q handler processing audio: %s", audio.file_id)
                    try:
                        audio_file = await context.bot.get_file(audio.file_id)
                        dl_audio_data = await download_media(audio_file.file_path)
                        if dl_audio_data:
                            audio_data = dl_audio_data
                            audio_mime_type = audio.mime_type
                            image_data_list = []  # Clear images
                            logger.info(
                                "Audio %s downloaded for /q. MIME: %s",
                                audio.file_id,
                                audio_mime_type,
                            )
                        else:
                            logger.error(
                                "Failed to download audio %s for /q.", audio.file_id
                            )
                    except Exception:  # noqa: BLE001
                        logger.error("Error processing audio for /q", exc_info=True)

            # Photo or sticker processing (only if video and audio was not processed)
            if not video_data and not audio_data:
                if target_message_for_media.sticker:
                    try:
                        sticker_file = await context.bot.get_file(
                            target_message_for_media.sticker.file_id
                        )
                        img_bytes = await download_media(sticker_file.file_path)
                        if img_bytes:
                            image_data_list.append(img_bytes)
                            logger.info(
                                "Added sticker image to /q list from message %s.",
                                target_message_for_media.message_id,
                            )
                    except Exception as e:  # noqa: BLE001
                        logger.error(
                            "Error downloading sticker for /q: %s",
                            e,
                            exc_info=True,
                        )
                # Check for media group first
                elif target_message_for_media.media_group_id:
                    logger.info(
                        "Q handler: Handling media group with ID: %s",
                        target_message_for_media.media_group_id,
                    )
                    # A simple sleep might work for small groups if messages arrive close together.
                    await asyncio.sleep(1)
                    media_messages = context.bot_data.get(
                        target_message_for_media.media_group_id, []
                    )
                    # The current message might not be in bot_data yet, so we add it.
                    if target_message_for_media not in media_messages:
                        media_messages.append(target_message_for_media)

                    for msg in media_messages:
                        if msg.photo:
                            photo = msg.photo[-1]
                            try:
                                photo_file = await context.bot.get_file(photo.file_id)
                                img_bytes = await download_media(photo_file.file_path)
                                if img_bytes:
                                    image_data_list.append(img_bytes)
                                    logger.info(
                                        "Added image from media group to /q list from message %s.",
                                        msg.message_id,
                                    )
                            except Exception as e:  # noqa: BLE001
                                logger.error(
                                    "Error downloading image from media group for /q: %s",
                                    e,
                                    exc_info=True,
                                )
                elif target_message_for_media.photo:
                    # Handle single photo
                    photo_size = target_message_for_media.photo[-1]
                    try:
                        file = await context.bot.get_file(photo_size.file_id)
                        img_bytes = await download_media(file.file_path)
                        if img_bytes:
                            image_data_list.append(img_bytes)
                            logger.info(
                                "Added single image to /q list from message %s.",
                                target_message_for_media.message_id,
                            )
                    except Exception as e:  # noqa: BLE001
                        logger.error(
                            "Error downloading single image for /q: %s",
                            e,
                            exc_info=True,
                        )

        if not query and not image_data_list and not video_data and not audio_data:
            await update.effective_message.reply_text(
                "Please provide a question, reply to media, or caption media with /q."
            )
            return

        if not query:  # Default prompt if only media is present
            if video_data:
                query = "Please analyze this video."
            elif audio_data:
                query = "Please analyze this audio."
            elif image_data_list:
                query = "Please analyze these image(s)."

        # Extract Telegraph content if present in the main query
        
        if query:
            query, more_telegraph_contents = await extract_telegraph_urls_and_content(
                query, update.effective_message.entities
            )
            telegraph_contents.extend(more_telegraph_contents)

            query, more_twitter_contents = await extract_twitter_urls_and_content(
                query,
                update.effective_message.entities,
                source_text=message_entity_text,
            )
            if more_twitter_contents:
                twitter_contents.extend(more_twitter_contents)

        # Download Telegraph media if available
        if telegraph_contents:
            try:
                telegraph_images, telegraph_video, telegraph_video_mime = (
                    await download_telegraph_media(telegraph_contents)
                )
                if telegraph_images:
                    image_data_list.extend(telegraph_images)
                    logger.info(
                        "Added %s images from Telegraph content to q_handler",
                        len(telegraph_images),
                    )
                if (
                    telegraph_video and not video_data
                ):  # Only use Telegraph video if no message video
                    video_data = telegraph_video
                    video_mime_type = telegraph_video_mime
                    logger.info(
                        "Added video from Telegraph content to q_handler with MIME: %s",
                        video_mime_type,
                    )
            except Exception as e:
                logger.error("Error downloading Telegraph media for q_handler: %s", e)


        if twitter_contents:
            try:
                twitter_images, twitter_video, twitter_video_mime = (
                    await download_twitter_media(twitter_contents)
                )
                if twitter_images:
                    image_data_list.extend(twitter_images)
                    logger.info(
                        "Added %s images from Twitter content to q_handler",
                        len(twitter_images),
                    )
                if twitter_video and not video_data:
                    video_data = twitter_video
                    video_mime_type = twitter_video_mime
                    logger.info(
                        "Added video from Twitter content to q_handler with MIME: %s",
                        video_mime_type,
                    )
            except Exception as e:
                logger.error("Error downloading Twitter media for q_handler: %s", e)

        if (
            query and not video_data and not image_data_list and not audio_data
        ):  # If query is present and no media is present, extract YouTube URLs
            # Extract YouTube URLs and replace in text
            query, youtube_urls = extract_youtube_urls(query)

        # Check if a specific model is requested or if OpenRouter is not available
        if call_model is not None or not is_openrouter_available():
            # Specific model requested or OpenRouter is disabled - process directly
            processing_message_text = "Processing your question..."
            if model_name:
                processing_message_text = f"Processing your question with {model_name}..."
            
            if video_data:
                processing_message_text = processing_message_text.replace("Processing", "Analyzing video and processing")
            elif audio_data:
                processing_message_text = processing_message_text.replace("Processing", "Analyzing audio and processing")
            elif image_data_list:
                processing_message_text = processing_message_text.replace("Processing", f"Analyzing {len(image_data_list)} image(s) and processing")
            elif twitter_contents:
                processing_message_text = processing_message_text.replace("Processing", f"Analyzing {len(twitter_contents)} Twitter post(s) and processing")
            elif youtube_urls and len(youtube_urls) > 0:
                processing_message_text = processing_message_text.replace("Processing", f"Analyzing {len(youtube_urls)} YouTube video(s) and processing")

            processing_message = await update.effective_message.reply_text(
                processing_message_text
            )
            
            # Process directly with specified model or Gemini
            if call_model is not None:
                await process_q_request_with_specific_model(
                    update, processing_message, query, original_query, 
                    image_data_list, video_data, video_mime_type,
                    audio_data, audio_mime_type, youtube_urls,
                    call_model, model_name
                )
            else:
                # OpenRouter not available, use Gemini
                await process_q_request_with_gemini(
                    update, processing_message, query, original_query, 
                    image_data_list, video_data, video_mime_type,
                    audio_data, audio_mime_type, youtube_urls
                )
            return
        
        # If the request contains video or YouTube links, process with Gemini directly.
        if video_data or (youtube_urls and len(youtube_urls) > 0):
            if video_data:
                processing_message_text = "Analyzing video and processing your question..."
            else:
                processing_message_text = "Analyzing linked YouTube video(s) and processing your question..."

            processing_message = await update.effective_message.reply_text(
                processing_message_text
            )

            await process_q_request_with_gemini(
                update,
                processing_message,
                query,
                original_query,
                image_data_list,
                video_data,
                video_mime_type,
                audio_data,
                audio_mime_type,
                youtube_urls,
            )
            return

        # OpenRouter is available - show model selection
        # Check if media is present to determine which models to show
        has_images = bool(image_data_list)
        has_av_media = bool(video_data or audio_data)
        has_media = has_images or has_av_media

        # Create model selection keyboard
        keyboard = create_model_selection_keyboard(
            has_media=has_media,
            has_images=has_images,
            has_av_media=has_av_media,
        )
        
        # Create selection message text
        selection_text = "Please select which AI model to use for your question:"
        if has_media:
            selection_text += "\n\n*Note: Only models that support media are shown.*"
        
        # Send model selection message
        selection_message = await update.effective_message.reply_text(
            selection_text,
            reply_markup=keyboard,
            parse_mode=ParseMode.MARKDOWN
        )

        if not language:
            language = languages.get(alpha_2=langid.classify(query)[0]).name

        username = "Anonymous"
        if update.effective_sender:  # Should be update.effective_message.from_user
            sender = update.effective_message.from_user
            if sender.full_name:
                username = sender.full_name
            elif sender.first_name and sender.last_name:
                username = f"{sender.first_name} {sender.last_name}"
            elif sender.first_name:
                username = sender.first_name
            elif sender.username:
                username = sender.username

        # For database logging, use the original query without Telegraph content expansion
        db_query_text = original_query
        if update.effective_message.reply_to_message:
            replied_text_original = (
                update.effective_message.reply_to_message.text
                or update.effective_message.reply_to_message.caption
                or ""
            )
            if replied_text_original:
                if db_query_text:
                    db_query_text = f'Context from replied message: "{replied_text_original}"\n\nQuestion: {db_query_text}'
                else:
                    db_query_text = replied_text_original

        # Store request context for later processing after model selection
        request_key = f"{update.effective_chat.id}_{update.effective_message.from_user.id}_{selection_message.message_id}"
        pending_q_requests[request_key] = {
            "user_id": update.effective_message.from_user.id,
            "username": username,
            "query": query,
            "original_query": original_query,
            "db_query_text": db_query_text,
            "language": language,
            "image_data_list": image_data_list,
            "video_data": video_data,
            "video_mime_type": video_mime_type,
            "audio_data": audio_data,
            "audio_mime_type": audio_mime_type,
            "youtube_urls": youtube_urls,
            "telegraph_contents": telegraph_contents,
            "twitter_contents": twitter_contents,
            "date": update.effective_message.date,
            "reply_to_message_id": (
                update.effective_message.reply_to_message.message_id
                if update.effective_message.reply_to_message
                else None
            ),
            "chat_id": update.effective_chat.id,
            "message_id": update.effective_message.message_id,
            "selection_message_id": selection_message.message_id,
            "original_user_id": update.effective_message.from_user.id,  # Store original user for validation
            "timestamp": time.time()  # For timeout handling
        }

        # Schedule default model processing if no selection is made
        asyncio.create_task(handle_model_timeout(request_key, context.bot))

        # Log database entry
        await queue_message_insert(
            user_id=update.effective_message.from_user.id,
            username=username,
            text=f"Ask {TELEGRAPH_AUTHOR_NAME}: {db_query_text}",
            language=language,
            date=update.effective_message.date,
            reply_to_message_id=(
                update.effective_message.reply_to_message.message_id
                if update.effective_message.reply_to_message
                else None
            ),
            chat_id=update.effective_chat.id,
            message_id=update.effective_message.message_id,
        )
    except Exception as e:  # noqa: BLE001
        logger.error("Error in q_handler: %s", e, exc_info=True)
        try:
            await update.effective_message.reply_text(
                f"Error processing your question: {str(e)}"
            )
        except Exception:  # noqa: BLE001 # Inner exception during error reporting
            pass


async def model_selection_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle model selection callback from inline keyboard."""
    query = update.callback_query
    if not query or not query.data:
        return

    # Answer the callback query to remove loading state
    await query.answer()

    # Check if this is a model selection callback
    if not query.data.startswith(MODEL_CALLBACK_PREFIX):
        return

    # Extract selected model
    selected_model = query.data[len(MODEL_CALLBACK_PREFIX):]
    
    # Get request key from message info
    chat_id = query.message.chat.id
    message_id = query.message.message_id
    user_id = query.from_user.id
    
    # Find the matching request
    request_key = None
    request_data = None
    for key, data in pending_q_requests.items():
        if (data["chat_id"] == chat_id and 
            data["selection_message_id"] == message_id):
            request_key = key
            request_data = data
            break
    
    if not request_data:
        await query.edit_message_text("Oops! This request has vanished into the void. Maybe it went for a coffee break ☕️.")
        return
    
    if not is_model_configured(selected_model):
        await query.answer("That model is not configured for this bot yet.", show_alert=True)
        return
    
    # Check if the user clicking is the original requester
    if request_data["original_user_id"] != user_id:
        await query.answer("Hey! No cookie stealing allowed. Only the original baker can eat this one. 🍪", show_alert=True)
        return
    
    # Check timeout (configurable via MODEL_SELECTION_TIMEOUT)
    if time.time() - request_data["timestamp"] > MODEL_SELECTION_TIMEOUT:
        # Remove expired request
        del pending_q_requests[request_key]
        await query.edit_message_text("Too slow! This request has turned into a pumpkin. Please try again before midnight 🎃.")
        return
    
    # Remove the request from pending list
    del pending_q_requests[request_key]
    
    try:
        # Update the selection message to show processing
        processing_text = f"Processing your question with {get_model_display_name(selected_model)}..."
        if request_data.get("video_data"):
            processing_text = f"Analyzing video and processing your question with {get_model_display_name(selected_model)}..."
        elif request_data.get("audio_data"):
            processing_text = f"Analyzing audio and processing your question with {get_model_display_name(selected_model)}..."
        elif request_data.get("image_data_list"):
            processing_text = f"Analyzing {len(request_data['image_data_list'])} image(s) and processing your question with {get_model_display_name(selected_model)}..."
        elif request_data.get("twitter_contents"):
            processing_text = f"Analyzing {len(request_data['twitter_contents'])} Twitter post(s) and processing your question with {get_model_display_name(selected_model)}..."
        elif request_data.get("youtube_urls") and len(request_data["youtube_urls"]) > 0:
            processing_text = f"Analyzing {len(request_data['youtube_urls'])} YouTube video(s) and processing your question with {get_model_display_name(selected_model)}..."
        
        await query.edit_message_text(processing_text)
        
        # Get model function and name based on selection
        call_model, model_name = get_model_function_and_name(selected_model)
        
        # Prepare system prompt
        current_datetime = datetime.utcnow().strftime("%H:%M:%S %B %d, %Y")
        system_prompt = Q_SYSTEM_PROMPT.format(
            current_datetime=current_datetime, 
            language=request_data["language"]
        )
        
        # Determine if pro model should be used
        capabilities = MODEL_CAPABILITIES.get(selected_model, {})
        supports_images = capabilities.get("images", False)
        supports_video = capabilities.get("video", False)
        supports_audio = capabilities.get("audio", False)

        image_data_list = request_data.get("image_data_list") if supports_images else None
        video_data = request_data.get("video_data") if supports_video else None
        audio_data = request_data.get("audio_data") if supports_audio else None
        video_mime_type = request_data.get("video_mime_type") if supports_video else None
        audio_mime_type = request_data.get("audio_mime_type") if supports_audio else None

        use_pro_model = bool(
            video_data
            or (image_data_list and len(image_data_list) > 0)
            or audio_data
            or request_data.get("youtube_urls")
        )
        
        # Call the selected model
        response = await call_model(
            system_prompt=system_prompt,
            user_content=request_data["query"],
            image_data_list=image_data_list,
            video_data=video_data,
            video_mime_type=video_mime_type,
            use_pro_model=use_pro_model,
            youtube_urls=request_data.get("youtube_urls"),
            audio_data=audio_data,
            audio_mime_type=audio_mime_type,
        )
        
        if response:
            if model_name == GPT_MODEL and isinstance(response, dict):
                analysis = response.get("analysis")
                final = response.get("final") or ""
                resp_text = ""
                if analysis:
                    resp_text += f"*Thought process*\n{analysis}\n\n"
                resp_text += f"*Final answer*\n{final}"
            else:
                resp_text = response if isinstance(response, str) else response.get("final", "")
            
            if model_name:
                resp_text = f"{resp_text}\n\n_Model: {model_name}_"
            
            # Send response using the processing message
            await send_response(
                query.message,
                resp_text,
                "Answer to Your Question",
                ParseMode.MARKDOWN,
            )
        else:
            await query.edit_message_text(
                "I couldn't find an answer to your question. Please try rephrasing or asking something else."
            )
            
    except Exception as e:
        logger.error("Error in model_selection_callback: %s", e, exc_info=True)
        try:
            await query.edit_message_text(f"Error processing your question: {str(e)}")
        except Exception:
            pass


def get_model_display_name(model_key: str) -> str:
    """Get display name for model key."""
    display_names = {
        MODEL_GEMINI: "Gemini 2.5 ✨",
        MODEL_LLAMA: "Llama 4",
        MODEL_GROK: "Grok 4",
        MODEL_QWEN: "Qwen 3",
        MODEL_DEEPSEEK: "DeepSeek 3.1"
    }
    return display_names.get(model_key, model_key)


def get_model_function_and_name(model_key: str) -> tuple:
    """Get model function and name for the given model key."""
    if model_key == MODEL_GEMINI:
        return call_gemini, None  # Default Gemini model
    elif model_key == MODEL_LLAMA and is_model_configured(MODEL_LLAMA):
        return partial(call_openrouter, model_name=LLAMA_MODEL), LLAMA_MODEL
    elif model_key == MODEL_GROK and is_model_configured(MODEL_GROK):
        return partial(call_openrouter, model_name=GROK_MODEL), GROK_MODEL
    elif model_key == MODEL_QWEN and is_model_configured(MODEL_QWEN):
        return partial(call_openrouter, model_name=QWEN_MODEL), QWEN_MODEL
    elif model_key == MODEL_DEEPSEEK and is_model_configured(MODEL_DEEPSEEK):
        return partial(call_openrouter, model_name=DEEPSEEK_MODEL), DEEPSEEK_MODEL
    else:
        return call_gemini, None


async def cleanup_expired_requests(bot=None):
    """Clean up expired pending Q requests (older than MODEL_SELECTION_TIMEOUT seconds) and delete messages."""
    current_time = time.time()
    expired_keys = []
    
    for key, request_data in pending_q_requests.items():
        if current_time - request_data["timestamp"] > MODEL_SELECTION_TIMEOUT:  # Configurable timeout
            expired_keys.append((key, request_data))
    
    for key, request_data in expired_keys:
        try:
            # Delete the selection message if bot is available
            if bot and "chat_id" in request_data and "selection_message_id" in request_data:
                await bot.delete_message(
                    chat_id=request_data["chat_id"],
                    message_id=request_data["selection_message_id"]
                )
                logger.info("Deleted expired model selection message: %s", request_data['selection_message_id'])
        except Exception as e:
            logger.warning("Failed to delete expired message %s: %s", request_data.get('selection_message_id'), e)
        
        # Remove from pending requests
        del pending_q_requests[key]
        logger.info("Cleaned up expired Q request: %s", key)
    
    return len(expired_keys)


async def handle_model_timeout(request_key: str, bot) -> None:
    """Process pending /q request with default model if user doesn't select a model."""
    await asyncio.sleep(MODEL_SELECTION_TIMEOUT)

    request_data = pending_q_requests.get(request_key)
    if not request_data:
        return

    # Remove the request from pending list
    del pending_q_requests[request_key]

    try:
        processing_text = (
            f"No model selected in time. Using {get_model_display_name(DEFAULT_Q_MODEL)}..."
        )
        processing_message = await bot.edit_message_text(
            chat_id=request_data["chat_id"],
            message_id=request_data["selection_message_id"],
            text=processing_text,
        )

        call_model, model_name = get_model_function_and_name(DEFAULT_Q_MODEL)

        current_datetime = datetime.utcnow().strftime("%H:%M:%S %B %d, %Y")
        system_prompt = Q_SYSTEM_PROMPT.format(
            current_datetime=current_datetime,
            language=request_data["language"],
        )

        capability_key = (
            DEFAULT_Q_MODEL
            if DEFAULT_Q_MODEL == MODEL_GEMINI or is_model_configured(DEFAULT_Q_MODEL)
            else MODEL_GEMINI
        )
        capabilities = MODEL_CAPABILITIES.get(
            capability_key, MODEL_CAPABILITIES[MODEL_GEMINI]
        )
        supports_images = capabilities.get("images", False)
        supports_video = capabilities.get("video", False)
        supports_audio = capabilities.get("audio", False)

        image_data_list = request_data.get("image_data_list") if supports_images else None
        video_data = request_data.get("video_data") if supports_video else None
        audio_data = request_data.get("audio_data") if supports_audio else None
        video_mime_type = request_data.get("video_mime_type") if supports_video else None
        audio_mime_type = request_data.get("audio_mime_type") if supports_audio else None

        use_pro_model = bool(
            video_data
            or (image_data_list and len(image_data_list) > 0)
            or audio_data
            or request_data.get("youtube_urls")
        )

        response = await call_model(
            system_prompt=system_prompt,
            user_content=request_data["query"],
            image_data_list=image_data_list,
            video_data=video_data,
            video_mime_type=video_mime_type,
            use_pro_model=use_pro_model,
            youtube_urls=request_data.get("youtube_urls"),
            audio_data=audio_data,
            audio_mime_type=audio_mime_type,
        )

        if response:
            if model_name == GPT_MODEL and isinstance(response, dict):
                analysis = response.get("analysis")
                final = response.get("final") or ""
                resp_text = ""
                if analysis:
                    resp_text += f"*Thought process*\n{analysis}\n\n"
                resp_text += f"*Final answer*\n{final}"
            else:
                resp_text = (
                    response if isinstance(response, str) else response.get("final", "")
                )

            if model_name:
                resp_text = f"{resp_text}\n\n_Model: {model_name}_"

            await send_response(
                processing_message,
                resp_text,
                "Answer to Your Question",
                ParseMode.MARKDOWN,
            )
        else:
            await bot.edit_message_text(
                chat_id=request_data["chat_id"],
                message_id=request_data["selection_message_id"],
                text=(
                    "I couldn't find an answer to your question. Please try rephrasing or "
                    "asking something else."
                ),
            )
    except Exception as e:  # noqa: BLE001
        logger.error("Error processing default model for /q: %s", e, exc_info=True)
        try:
            await bot.edit_message_text(
                chat_id=request_data["chat_id"],
                message_id=request_data["selection_message_id"],
                text="Error processing your question.",
            )
        except Exception:  # noqa: BLE001
            pass


async def periodic_cleanup_task(bot):
    """Periodically clean up expired requests every 5 seconds."""
    while True:
        try:
            await asyncio.sleep(5)  # Wait 5 seconds
            cleaned_count = await cleanup_expired_requests(bot)
            if cleaned_count > 0:
                logger.debug("Periodic cleanup: removed %s expired requests", cleaned_count)
        except Exception as e:
            logger.error("Error in periodic cleanup task: %s", e, exc_info=True)


async def start_periodic_cleanup(bot):
    """Start the periodic cleanup task."""
    global _cleanup_task
    if _cleanup_task is None or _cleanup_task.done():
        _cleanup_task = asyncio.create_task(periodic_cleanup_task(bot))
        logger.info("Started periodic cleanup task for model selection timeouts")


async def stop_periodic_cleanup():
    """Stop the periodic cleanup task."""
    global _cleanup_task
    if _cleanup_task and not _cleanup_task.done():
        _cleanup_task.cancel()
        try:
            await _cleanup_task
        except asyncio.CancelledError:
            pass
        logger.info("Stopped periodic cleanup task")


async def deepseek_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /deepseek command."""
    await q_handler(
        update,
        context,
        call_model=partial(call_openrouter, model_name=DEEPSEEK_MODEL),
        model_name=DEEPSEEK_MODEL,
    )


async def qwen_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /qwen command."""
    await q_handler(
        update,
        context,
        call_model=partial(call_openrouter, model_name=QWEN_MODEL),
        model_name=QWEN_MODEL,
    )


async def llama_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /llama command."""
    await q_handler(
        update,
        context,
        call_model=partial(call_openrouter, model_name=LLAMA_MODEL),
        model_name=LLAMA_MODEL,
    )


async def gpt_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /gpt command."""
    await q_handler(
        update,
        context,
        call_model=partial(call_openrouter, model_name=GPT_MODEL),
        model_name=GPT_MODEL,
    )
