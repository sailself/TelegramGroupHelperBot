"""High-level command handlers for TelegramGroupHelperBot."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from io import BytesIO
from typing import List, Optional

import langid
from pycountry import languages
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, InputMediaPhoto, Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from bot.config import (
    FACTCHECK_SYSTEM_PROMPT,
    GEMINI_IMAGE_MODEL,
    PAINTME_SYSTEM_PROMPT,
    PORTRAIT_SYSTEM_PROMPT,
    PROFILEME_SYSTEM_PROMPT,
    SUPPORT_LINK,
    SUPPORT_MESSAGE,
    TELEGRAPH_AUTHOR_NAME,
    TLDR_SYSTEM_PROMPT,
    USE_VERTEX_IMAGE,
    USER_HISTORY_MESSAGE_COUNT,
    VERTEX_IMAGE_MODEL,
)
from bot.db.database import (
    queue_message_insert,
    select_messages_by_user,
    select_messages_from_id,
)
from bot.llm import (
    ImageGenerationError,
    call_gemini,
    generate_image_with_gemini,
    generate_image_with_vertex,
    generate_video_with_veo,
    download_media,
)

from .access import check_access_control, is_rate_limited
from .content import (
    create_telegraph_page,
    download_telegraph_media,
    download_twitter_media,
    extract_telegraph_urls_and_content,
    extract_twitter_urls_and_content,
    extract_youtube_urls,
)
from .responses import send_response

logger = logging.getLogger(__name__)


async def tldr_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /tldr command.

    Args:
        update: The update containing the message.
        context: The context object.
    """
    if update.effective_message is None or update.effective_chat is None:
        return

    # Check access control
    if not await check_access_control(update, "tldr"):
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
        messages = await select_messages_from_id(
            chat_id=update.effective_chat.id, message_id=from_message_id
        )

        if not messages or len(messages) == 0:
            await processing_message.edit_text("No messages found to summarize.")
            return

        # Format messages for the LLM
        formatted_messages = "Recent conversation:\n\n"
        for msg in messages:
            timestamp = msg.date.strftime("%Y-%m-%d %H:%M:%S")
            # Use the username if available, otherwise use "Anonymous" as fallback
            username = (
                msg.username if msg.username and msg.username.strip() else "Anonymous"
            )
            if msg.user_id:
                username = f"{username} (id:{msg.user_id})"
            if msg.reply_to_message_id:
                formatted_messages += f"msg[{msg.message_id}] reply_to[{msg.reply_to_message_id}] {timestamp} - {username}: {msg.text}\n\n"
            else:
                formatted_messages += (
                    f"msg[{msg.message_id}] {timestamp} - {username}: {msg.text}\n\n"
                )

        # Use the configured system prompt
        prompt = TLDR_SYSTEM_PROMPT.format(bot_name=TELEGRAPH_AUTHOR_NAME)

        # Generate summary using Gemini
        response = await call_gemini(
            system_prompt=prompt, user_content=formatted_messages
        )

        if response:
            await send_response(
                processing_message, response, "Message Summary", ParseMode.MARKDOWN
            )
        else:
            await processing_message.edit_text(
                "Failed to generate a summary. Please try again later."
            )

    except Exception as e:  # noqa: BLE001
        logger.error("Error in tldr_handler: %s", e, exc_info=True)
        try:
            await update.effective_message.reply_text(
                f"Error summarizing messages: {str(e)}"
            )
        except Exception:  # noqa: BLE001
            pass


async def factcheck_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /factcheck command.

    Args:
        update: The update containing the message.
        context: The context object.
    """
    if update.effective_message is None or update.effective_chat is None:
        return

    # Check access control
    if not await check_access_control(update, "factcheck"):
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
    replied_message = update.effective_message.reply_to_message
    original_message_text = replied_message.text or replied_message.caption or ""

    # Extract Telegraph content if present (for AI processing)
    message_to_check = original_message_text
    telegraph_contents = []
    twitter_contents = []
    if message_to_check:
        message_to_check, telegraph_contents = await extract_telegraph_urls_and_content(
            message_to_check, replied_message.entities
        )
        message_to_check, twitter_contents = await extract_twitter_urls_and_content(
            message_to_check,
            replied_message.entities,
            source_text=original_message_text,
        )

    image_data_list: List[bytes] = []
    video_data: Optional[bytes] = None
    video_mime_type: Optional[str] = None
    audio_data: Optional[bytes] = None
    audio_mime_type: Optional[str] = None
    youtube_urls = None

    # Download Telegraph media if available
    if telegraph_contents:
        try:
            telegraph_images, telegraph_video, telegraph_video_mime = (
                await download_telegraph_media(telegraph_contents)
            )
            if telegraph_images:
                image_data_list.extend(telegraph_images)
                logger.info(
                    "Added %s images from Telegraph content to factcheck",
                    len(telegraph_images),
                )
            if (
                telegraph_video and not video_data
            ):  # Only use Telegraph video if no message video
                video_data = telegraph_video
                video_mime_type = telegraph_video_mime
                logger.info(
                    "Added video from Telegraph content to factcheck with MIME: %s",
                    video_mime_type,
                )
        except Exception as e:
            logger.error("Error downloading Telegraph media for factcheck: %s", e)


    if twitter_contents:
        try:
            twitter_images, twitter_video, twitter_video_mime = (
                await download_twitter_media(twitter_contents)
            )
            if twitter_images:
                image_data_list.extend(twitter_images)
                logger.info(
                    "Added %s images from Twitter content to factcheck",
                    len(twitter_images),
                )
            if twitter_video and not video_data:
                video_data = twitter_video
                video_mime_type = twitter_video_mime
                logger.info(
                    "Added video from Twitter content to factcheck with MIME: %s",
                    video_mime_type,
                )
        except Exception as e:
            logger.error("Error downloading Twitter media for factcheck: %s", e)

    # Video processing (takes precedence)
    if replied_message.video:
        logger.info(
            "Fact-checking video: %s in chat %s",
            replied_message.video.file_id,
            replied_message.chat.id,
        )
        try:
            video_file = await context.bot.get_file(replied_message.video.file_id)
            video_mime_type = replied_message.video.mime_type
            dl_video_data = await download_media(video_file.file_path)
            if dl_video_data:
                video_data = dl_video_data
                image_data_list = []  # Clear images if video is present
                logger.info(
                    "Video %s downloaded for fact-check. MIME: %s",
                    replied_message.video.file_id,
                    video_mime_type,
                )
                if not message_to_check:  # If no caption, use default prompt for video
                    message_to_check = "Please fact-check this video."
            else:
                logger.error(
                    "Failed to download video %s for fact-check.",
                    replied_message.video.file_id,
                )
        except Exception as e:  # noqa: BLE001
            logger.error("Error processing video for fact-check: %s", e, exc_info=True)

    # Audio processing (takes precedence)
    if not video_data:
        audio = (
            replied_message.audio if replied_message.audio else replied_message.voice
        )
        if audio:
            logger.info("Fact-checking audio: %s", audio.file_id)
            try:
                audio_file = await context.bot.get_file(audio.file_id)
                dl_audio_data = await download_media(audio_file.file_path)
                if dl_audio_data:
                    audio_data = dl_audio_data
                    audio_mime_type = audio.mime_type
                    image_data_list = []  # Clear images
                    logger.info(
                        "Audio %s downloaded for fact-check. MIME: %s",
                        audio.file_id,
                        audio_mime_type,
                    )
                else:
                    logger.error(
                        "Failed to download audio %s for fact-check.",
                        audio.file_id,
                    )
            except Exception as e:  # noqa: BLE001
                logger.error(
                    "Error processing audio for fact-check: %s", e, exc_info=True
                )

    # Photo or sticker processing (only if video and audio were not processed)
    if not video_data and not audio_data:
        if replied_message.sticker:
            try:
                sticker_file = await context.bot.get_file(replied_message.sticker.file_id)
                img_bytes = await download_media(sticker_file.file_path)
                if img_bytes:
                    image_data_list.append(img_bytes)
                    logger.info(
                        "Added sticker image to fact-check list from message %s.",
                        replied_message.message_id,
                    )
            except Exception:  # noqa: BLE001
                logger.error(
                    "Error downloading sticker for fact-check", exc_info=True
                )
        elif replied_message.photo:
            # Simplified to handle only the single photo from the replied message
            photo_size = replied_message.photo[-1]
            try:
                file = await context.bot.get_file(photo_size.file_id)
                img_bytes = await download_media(file.file_path)
                if img_bytes:
                    image_data_list.append(img_bytes)
                    logger.info(
                        "Added single image to fact-check list from message %s.",
                        replied_message.message_id,
                    )
            except Exception:  # noqa: BLE001
                logger.error(
                    "Error downloading single image for fact-check", exc_info=True
                )

    if (
        not message_to_check
        and not image_data_list
        and not video_data
        and not audio_data
    ):
        await update.effective_message.reply_text(
            "Cannot fact-check an empty message with no media (image/video/audio)."
        )
        return
    if (
        message_to_check and not video_data and not image_data_list and not audio_data
    ):  # If message is present and no media is present, extract YouTube URLs
        # Extract YouTube URLs and replace in text
        message_to_check, youtube_urls = extract_youtube_urls(message_to_check)

    # Default prompt if only media is present
    if not message_to_check:
        if video_data:
            message_to_check = "Please analyze this video and verify any claims or content shown in it."
        elif audio_data:
            message_to_check = "Please analyze this audio and verify any claims or content shown in it."
        elif image_data_list:
            message_to_check = "Please analyze these images and verify any claims or content shown in them."

    language = languages.get(alpha_2=langid.classify(message_to_check)[0]).name

    processing_message_text = "Fact-checking message..."
    if video_data:
        processing_message_text = "Analyzing video and fact-checking content..."
    elif audio_data:
        processing_message_text = "Analyzing audio and fact-checking content..."
    elif image_data_list:
        processing_message_text = (
            f"Analyzing {len(image_data_list)} image(s) and fact-checking content..."
        )
    elif youtube_urls and len(youtube_urls) > 0:
        processing_message_text = f"Analyzing {len(youtube_urls)} YouTube video(s) and fact-checking content..."

    if telegraph_contents:
        telegraph_media_info = ""
        if any(content.get("image_urls") for content in telegraph_contents):
            image_count = sum(
                len(content.get("image_urls", [])) for content in telegraph_contents
            )
            telegraph_media_info += f" with {image_count} image(s)"
        if any(content.get("video_urls") for content in telegraph_contents):
            video_count = sum(
                len(content.get("video_urls", [])) for content in telegraph_contents
            )
            telegraph_media_info += (
                f" and {video_count} video(s)"
                if telegraph_media_info
                else f" with {video_count} video(s)"
            )

        if processing_message_text == "Fact-checking message...":
            processing_message_text = f"Extracting and fact-checking content from {len(telegraph_contents)} Telegraph page(s){telegraph_media_info}..."
        else:
            processing_message_text = f"{processing_message_text[:-3]} and {len(telegraph_contents)} Telegraph page(s){telegraph_media_info}..."


    if twitter_contents:
        twitter_media_info = ""
        if any(content.get("image_urls") for content in twitter_contents):
            image_count = sum(
                len(content.get("image_urls", [])) for content in twitter_contents
            )
            twitter_media_info += f" with {image_count} image(s)"
        if any(content.get("video_urls") for content in twitter_contents):
            video_count = sum(
                len(content.get("video_urls", [])) for content in twitter_contents
            )
            twitter_media_info += (
                f" and {video_count} video(s)"
                if twitter_media_info
                else f" with {video_count} video(s)"
            )
    
        if processing_message_text == "Fact-checking message...":
            processing_message_text = (
                f"Extracting and fact-checking content from {len(twitter_contents)} Twitter post(s)"
                f"{twitter_media_info}..."
            )
        else:
            processing_message_text = (
                f"{processing_message_text[:-3]} and {len(twitter_contents)} Twitter post(s)"
                f"{twitter_media_info}..."
            )

    processing_message = await update.effective_message.reply_text(
        processing_message_text
    )

    try:
        # Format the system prompt with the current date
        current_datetime = datetime.utcnow().strftime("%H:%M:%S %B %d, %Y")
        system_prompt = FACTCHECK_SYSTEM_PROMPT.format(
            current_datetime=current_datetime
        )
        use_pro_model = bool(
            video_data or image_data_list or audio_data
        )  # Use Pro model if media is present

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
            audio_data=audio_data,
            audio_mime_type=audio_mime_type,
        )

        resp_msg = "Checking facts"
        if video_data:
            resp_msg = "Analyzing video and checking facts"
        elif audio_data:
            resp_msg = "Analyzing audio and checking facts"
        elif image_data_list:
            resp_msg = f"Analyzing {len(image_data_list)} image(s) and checking facts"

        if use_pro_model:
            resp_msg = f"{resp_msg} with Pro Model...\nIt could take longer than usual, please be patient."
        else:
            resp_msg = f"{resp_msg}..."

        await processing_message.edit_text(resp_msg, parse_mode=None)

        # Final update with complete response
        if full_response:
            await send_response(
                processing_message,
                full_response,
                "Fact Check Results",
                ParseMode.MARKDOWN,
            )
        else:
            await processing_message.edit_text(
                "Failed to fact-check the message. Please try again later."
            )

    except Exception as e:  # noqa: BLE001
        logger.error("Error in factcheck_handler: %s", e, exc_info=True)
        try:
            await processing_message.edit_text(f"Error fact-checking message: {str(e)}")
        except Exception:  # noqa: BLE001 # Inner exception during error reporting
            pass


async def handle_media_group(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle messages that are part of a media group."""
    message = update.effective_message
    if message.media_group_id:
        if message.media_group_id not in context.bot_data:
            context.bot_data[message.media_group_id] = [message]
        else:
            # To avoid duplicates if the handler is called for the same message multiple times
            if message not in context.bot_data[message.media_group_id]:
                context.bot_data[message.media_group_id].append(message)


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

    # Check access control
    if not await check_access_control(update, "img"):
        return

    # Check rate limiting
    if is_rate_limited(user_id):
        await update.effective_message.reply_text(
            "You're sending commands too quickly. Please wait a moment before trying again."
        )
        return

    # Get message text without the command
    original_message_text = (
        update.effective_message.text or update.effective_message.caption or ""
    )
    if original_message_text.startswith("/img"):
        prompt = original_message_text[4:].strip()
    else:
        prompt = ""

    image_urls = []

    # Check for media group
    if update.effective_message.media_group_id:
        # This part is tricky as we need to gather all messages from the media group.
        # We'll rely on a caching mechanism that should be implemented in the main bot logic.
        # For now, we'll assume a helper function get_media_group_messages exists.
        # This is a placeholder for a more robust implementation.
        logger.info(
            "Handling media group with ID: %s", update.effective_message.media_group_id
        )
        # A simple sleep might work for small groups if messages arrive close together.
        await asyncio.sleep(1)
        media_messages = context.bot_data.get(
            update.effective_message.media_group_id, []
        )
        # The current message might not be in bot_data yet, so we add it.
        if update.effective_message not in media_messages:
            media_messages.append(update.effective_message)

        for msg in media_messages:
            if msg.photo:
                photo = msg.photo[-1]
                photo_file = await context.bot.get_file(photo.file_id)
                image_urls.append(photo_file.file_path)
    elif update.effective_message.photo:
        photo = update.effective_message.photo[-1]
        photo_file = await context.bot.get_file(photo.file_id)
        image_urls.append(photo_file.file_path)
    elif update.effective_message.sticker:
        sticker_file = await context.bot.get_file(
            update.effective_message.sticker.file_id
        )
        image_urls.append(sticker_file.file_path)

    # Get potential image from a replied message if no image in the command message
    if not image_urls and update.effective_message.reply_to_message:
        replied_message = update.effective_message.reply_to_message
        # Check for media group
        if replied_message.media_group_id:
            # This part is tricky as we need to gather all messages from the media group.
            # We'll rely on a caching mechanism that should be implemented in the main bot logic.
            # For now, we'll assume a helper function get_media_group_messages exists.
            # This is a placeholder for a more robust implementation.
            logger.info(
                "Handling media group with ID: %s", replied_message.media_group_id
            )
            # A simple sleep might work for small groups if messages arrive close together.
            await asyncio.sleep(1)
            media_messages = context.bot_data.get(replied_message.media_group_id, [])
            # The current message might not be in bot_data yet, so we add it.
            if replied_message not in media_messages:
                media_messages.append(replied_message)

            for msg in media_messages:
                if msg.photo:
                    photo = msg.photo[-1]
                    photo_file = await context.bot.get_file(photo.file_id)
                    image_urls.append(photo_file.file_path)
        elif replied_message.photo:
            photo = replied_message.photo[-1]
            photo_file = await context.bot.get_file(photo.file_id)
            image_urls.append(photo_file.file_path)  # Add the photo to the list
        elif replied_message.sticker:
            sticker_file = await context.bot.get_file(
                replied_message.sticker.file_id
            )
            image_urls.append(sticker_file.file_path)

        replied_text_content = replied_message.text or replied_message.caption or ""
        if replied_text_content:
            # Extract Telegraph content if present in replied message
            replied_text_content, _ = await extract_telegraph_urls_and_content(
                replied_text_content, replied_message.entities
            )

            if prompt:
                if not image_urls:
                    prompt = f"{replied_text_content}\n\n{prompt}"
            else:
                prompt = replied_text_content

    if not prompt:
        await update.effective_message.reply_text(
            "Please provide a description of the image you want to generate or edit. "
            "For example: /img a cat playing piano"
        )
        return

    # Extract Telegraph content if present in the main prompt
    telegraph_contents = []
    if prompt:
        prompt, telegraph_contents = await extract_telegraph_urls_and_content(
            prompt, update.effective_message.entities
        )

    # Send a processing message
    processing_message_text = "Processing your image request... This may take a moment."
    if telegraph_contents:
        processing_message_text = f"Extracting content from {len(telegraph_contents)} Telegraph page(s) and processing your image request... This may take a moment."

    processing_message = await update.effective_message.reply_text(
        processing_message_text
    )

    try:
        # Check if we should use Vertex AI for text-only image generation
        if USE_VERTEX_IMAGE and not image_urls:
            # Use Vertex AI for text-only image generation
            logger.info(
                "img_handler operating in Vertex AI mode for text-only prompt: '%s'",
                prompt,
            )

            try:
                images_data_list = await generate_image_with_vertex(
                    prompt=prompt, number_of_images=1
                )
                image_data = images_data_list[0] if images_data_list else None
            except Exception as vertex_error:
                logger.error(
                    "Vertex AI image generation failed: %s", vertex_error, exc_info=True
                )
                # Fallback to Gemini if Vertex fails
                logger.info("Falling back to Gemini for image generation")
                image_data = await generate_image_with_gemini(
                    prompt=prompt, input_image_urls=image_urls
                )
        else:
            # Use Gemini for image editing (with input images) or when Vertex is disabled
            logger.info("img_handler operating in Gemini mode for prompt: '%s'", prompt)

            if image_urls:
                logger.info(
                    "Gemini: Processing image edit request with %d image(s): '%s'",
                    len(image_urls),
                    prompt,
                )
            else:
                logger.info("Gemini: Processing image generation request: '%s'", prompt)

            image_data = await generate_image_with_gemini(
                prompt=prompt, input_image_urls=image_urls
            )

        try:

            if image_data:
                # Determine which model was used based on the logic above
                model_used = (
                    VERTEX_IMAGE_MODEL
                    if (USE_VERTEX_IMAGE and not image_urls)
                    else GEMINI_IMAGE_MODEL
                )
                service_name = (
                    "Vertex AI" if (USE_VERTEX_IMAGE and not image_urls) else "Gemini"
                )

                logger.info(
                    "%s: Image data received. Attempting to send.", service_name
                )
                try:
                    image_io = BytesIO(image_data)
                    image_io.name = f"{service_name.lower()}_generated_image.jpg"
                    # Check if caption would be too long (> 1000 characters)
                    base_caption = f"Generated by *{model_used}*"
                    full_caption = f"{base_caption} with prompt: \n```\n{prompt}\n```"

                    if len(full_caption) > 1000:
                        # Create Telegraph page for the prompt
                        telegraph_url = await create_telegraph_page(
                            "Image Generation Prompt", f"{prompt}"
                        )
                        if telegraph_url:
                            caption = f"{base_caption} with prompt: \n[View it here]({telegraph_url})"
                        else:
                            # Fallback if Telegraph creation fails
                            truncated_prompt = (
                                prompt[:900] + "..." if len(prompt) > 900 else prompt
                            )
                            caption = f"{base_caption} with prompt: \n```\n{truncated_prompt}\n```"
                    else:
                        caption = full_caption

                    await processing_message.edit_media(
                        media=InputMediaPhoto(
                            media=image_io,
                            caption=caption,
                            parse_mode=ParseMode.MARKDOWN,
                        )
                    )
                    logger.info(
                        "%s: Image sent successfully by editing processing message with %s.",
                        service_name,
                        model_used,
                    )
                except Exception:  # noqa: BLE001
                    logger.error(
                        "%s: Error sending image via Telegram",
                        service_name,
                        exc_info=True,
                    )
            else:
                # Determine which service was attempted for error messages
                service_name = (
                    "Vertex AI" if (USE_VERTEX_IMAGE and not image_urls) else "Gemini"
                )
                logger.warning(
                    "%s: Image generation failed (image_data is None).", service_name
                )
                if image_urls:
                    await processing_message.edit_text(
                        f"I couldn't edit the image with {service_name}. Please try:\n"
                        "1. Simpler edit description\n2. More specific details\n3. Different edit type or try later"
                    )
                else:
                    await processing_message.edit_text(
                        f"I couldn't generate an image with {service_name}. Please try:\n"
                        "1. Simpler prompt\n2. More specific details\n3. Try again later"
                    )
        except ImageGenerationError as e:
            logger.error("ImageGenerationError in img_handler: %s", e)
            await processing_message.edit_text(f"Image generation failed: {e}")

        language = languages.get(alpha_2=langid.classify(original_message_text)[0]).name
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

        await queue_message_insert(
            user_id=update.effective_message.from_user.id,
            username=username,
            text=original_message_text,  # Use original text without Telegraph content expansion
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
        logger.error("Error in img_handler: %s", e, exc_info=True)
        error_message_for_user = (
            "Sorry, an unexpected error occurred while processing your image request."
        )
        try:
            await processing_message.edit_text(error_message_for_user)
        except Exception as edit_err:
            logger.error(
                "Failed to edit processing message with error: %s",
                edit_err,
                exc_info=True,
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

    # Check access control
    if not await check_access_control(update, "vid"):
        return

    # Rate Limiting
    if is_rate_limited(user_id):
        await update.effective_message.reply_text(
            "You're sending commands too quickly. Please wait a moment before trying again."
        )
        return

    logger.info("Received /vid command from user %s", user_id)

    # Prompt Extraction
    original_message_text = update.effective_message.text or ""
    prompt = ""
    if original_message_text.startswith("/vid"):
        prompt = original_message_text[len("/vid") :].strip()

    # Image Handling
    image_data_bytes: Optional[bytes] = None
    replied_message = update.effective_message.reply_to_message

    if replied_message:
        if replied_message.photo:
            logger.info("Replying to photo for /vid command.")
            photo = replied_message.photo[-1]  # Get the largest photo
            try:
                photo_file = await context.bot.get_file(photo.file_id)
                image_data_bytes = await download_media(photo_file.file_path)
                if image_data_bytes:
                    logger.info("Image downloaded successfully for /vid command.")
                else:
                    logger.warning("Failed to download image for /vid.")
            except Exception as e:  # noqa: BLE001
                logger.error("Error downloading image for /vid: %s", e, exc_info=True)
                await update.effective_message.reply_text(
                    "Error downloading image. Please try again."
                )
                return
        elif replied_message.sticker:
            logger.info("Replying to sticker for /vid command.")
            try:
                sticker_file = await context.bot.get_file(
                    replied_message.sticker.file_id
                )
                image_data_bytes = await download_media(sticker_file.file_path)
                if image_data_bytes:
                    logger.info("Sticker downloaded successfully for /vid command.")
                else:
                    logger.warning("Failed to download sticker for /vid.")
            except Exception as e:  # noqa: BLE001
                logger.error("Error downloading sticker for /vid: %s", e, exc_info=True)
                await update.effective_message.reply_text(
                    "Error downloading sticker. Please try again."
                )
                return

        # Use caption as prompt if no command prompt and caption exists
        if not prompt and replied_message.caption:
            prompt = replied_message.caption.strip()
            logger.info("Using caption from replied message as prompt: '{prompt}'")

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
        logger.info("Using default prompt for image-only /vid: '{prompt}'")

    processing_message = await update.effective_message.reply_text(
        "Processing video request... This may take a few minutes."
    )

    try:
        language = languages.get(alpha_2=langid.classify(original_message_text)[0]).name

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

        await queue_message_insert(
            user_id=update.effective_message.from_user.id,  # Use from_user here
            username=username,
            text=original_message_text,  # Use original text without Telegraph content expansion
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

        # Call Video Generation
        logger.info(
            "Calling generate_video_with_veo with prompt: '%s' and image_data: '%s'",
            prompt,
            "present" if image_data_bytes else "absent",
        )
        video_bytes, video_mime_type = await generate_video_with_veo(
            user_prompt=prompt, image_data=image_data_bytes
        )

        # Response Handling
        if video_bytes:
            logger.info(
                "Video generated successfully. MIME type: %s, Size: %d bytes",
                video_mime_type,
                len(video_bytes),
            )
            video_file = BytesIO(video_bytes)
            video_file.name = "generated_video.mp4"  # Suggested name

            try:
                await update.effective_message.reply_video(
                    video=video_file,
                    caption="Here's your generated video!",
                    read_timeout=120,  # Increased timeouts
                    write_timeout=120,
                    connect_timeout=60,
                    pool_timeout=60,
                )
                await processing_message.delete()
                logger.info("Video sent successfully and processing message deleted.")
            except Exception as e_telegram:
                logger.error(
                    "Error sending video via Telegram: %s", e_telegram, exc_info=True
                )
                await processing_message.edit_text(
                    "Sorry, I generated the video but couldn't send it via Telegram. It might be too large or in an unsupported format."
                )
        else:
            logger.warning("Video generation failed (video_bytes is None).")
            await processing_message.edit_text(
                "Sorry, I couldn't generate the video. Please try a different prompt or image. The model might have limitations or be unavailable."
            )

    except Exception as e:  # noqa: BLE001
        logger.error("Error in vid_handler: %s", e, exc_info=True)
        error_message_user = "Sorry, an unexpected error occurred while generating your video. Please try again later."
        if "timeout" in str(e).lower():
            error_message_user = "The video generation timed out. Please try a shorter prompt or a smaller image."
        elif "unsupported" in str(e).lower():
            error_message_user = "The video generation failed due to an unsupported format or feature. Please check your input."
        try:
            await processing_message.edit_text(error_message_user)
        except Exception:  # noqa: BLE001 # If editing fails, just log
            logger.error("Failed to edit processing message with error.", exc_info=True)


async def start_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:  # noqa: ARG001
    """Handle the /start command.

    Args:
        update: The update containing the message.
        context: The context object.
    """
    if update.effective_message is None or update.effective_chat is None:
        return

    # Check access control
    if not await check_access_control(update, "start"):
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
        "• /support - Show support information and Ko-fi link\n"
        "• /help - Show this help message\n\n"
        "Just type one of these commands to get started!"
    )

    await update.effective_message.reply_text(welcome_message)


async def paintme_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:  # noqa: ARG001
    """Handle the /paintme command.

    Fetches user's chat history, generates an image prompt with Gemini,
    then generates an image and sends it to the user.
    """
    if (
        update.effective_user is None
        or update.effective_message is None
        or update.effective_chat is None
    ):
        return

    user_id = update.effective_user.id

    # Check access control
    if not await check_access_control(update, "paintme"):
        return

    if is_rate_limited(user_id):
        await update.effective_message.reply_text(
            "Rate limit exceeded. Please try again later."
        )
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
            limit=USER_HISTORY_MESSAGE_COUNT,
        )

        if not user_messages or len(user_messages) < 5:  # Need a few messages at least
            await processing_message.edit_text(
                f"I don't have enough of your messages (at least 5 recent ones) in this chat to {reply_text_part}. Keep chatting!"
            )
            return

        # 2. Format chat history for Gemini prompt generation
        formatted_history = "User's recent messages in the group chat:\n\n"
        for msg in user_messages:
            formatted_history += f'- "{msg.text}"\n'

        # 3. Generate image prompt with Gemini
        image_prompt = await call_gemini(
            system_prompt=system_prompt,  # Use imported constant
            user_content=formatted_history,
            use_search_grounding=False,
        )

        if not image_prompt or "No response generated" in image_prompt:
            await processing_message.edit_text(
                "I couldn't come up with an image idea for you at this time. Please try again later."
            )
            return

        await processing_message.edit_text(
            f'Generated image prompt: "{image_prompt}". Now creating your masterpiece...'
        )

        # 4. Generate Image
        if USE_VERTEX_IMAGE:
            logger.info(
                "{'portrait' if isPortrait else 'paint'}me_handler: Using Vertex AI for image generation with prompt: '{image_prompt}'"
            )
            images_data_list = await generate_image_with_vertex(
                prompt=image_prompt, number_of_images=1
            )  # Generate 1 for this command

            if images_data_list and images_data_list[0]:
                image_data = images_data_list[0]
                image_io = BytesIO(image_data)
                image_io.name = "vertex_paintme_image.jpg"
                await update.effective_message.reply_photo(
                    photo=image_io,
                    caption=f'Here\'s your artistic representation!\nPrompt: "{image_prompt}"\nModel: {VERTEX_IMAGE_MODEL}',
                )
                await processing_message.delete()
            else:
                logger.error(
                    "{'portrait' if isPortrait else 'paint'}me_handler: Vertex AI image generation failed or returned no image."
                )
                await processing_message.edit_text(
                    f"Sorry, {VERTEX_IMAGE_MODEL} couldn't {reply_text_part}. Please try again."
                )

        else:  # Use Gemini for image generation
            logger.info(
                "{'portrait' if isPortrait else 'paint'}me_handler: Using Gemini for image generation with prompt: '{image_prompt}'"
            )
            # System prompt for Gemini image generation can be simple
            image_data = await generate_image_with_gemini(prompt=image_prompt)

            if image_data:
                image_io = BytesIO(image_data)
                image_io.name = "gemini_paintme_image.jpg"
                await update.effective_message.reply_photo(
                    photo=image_io,
                    caption=f'Here\'s your artistic representation!\nPrompt: "{image_prompt}"\nModel: Gemini',
                )
                await processing_message.delete()
            else:
                logger.warning(
                    "{'portrait' if isPortrait else 'paint'}me_handler: Gemini image generation failed."
                )
                await processing_message.edit_text(
                    f"Sorry, Gemini couldn't {reply_text_part}. Please try again."
                )

        # Log the command usage
        language = languages.get(
            alpha_2=langid.classify(formatted_history)[0]
        ).name  # Use history for language context
        username = "Anonymous"
        if update.effective_message.from_user:
            sender = update.effective_message.from_user
            if sender.full_name:
                username = sender.full_name
            elif sender.first_name:
                username = sender.first_name
            elif sender.username:
                username = sender.username

        await queue_message_insert(
            user_id=user_id,
            username=username,
            text=update.effective_message.text,  # Log the command itself
            language=language,  # Could be 'xx' if history is very short/non-textual
            date=update.effective_message.date,
            chat_id=update.effective_chat.id,
            message_id=update.effective_message.message_id,
        )

    except Exception as e:  # noqa: BLE001
        logger.error("Error in paintme_handler: %s", e, exc_info=True)
        try:
            await processing_message.edit_text(
                f"Sorry, an unexpected error occurred while {'creating your portrait' if isPortrait else 'painting your picture'}: {str(e)}"
            )
        except Exception:  # noqa: BLE001
            pass


async def profileme_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:  # noqa: ARG001
    """Handle the /profileme command.

    Args:
        update: The update containing the message.
        context: The context object.
    """
    if (
        update.effective_user is None
        or update.effective_message is None
        or update.effective_chat is None
    ):
        return

    user_id = update.effective_user.id

    # Check access control
    if not await check_access_control(update, "profileme"):
        return

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
            limit=USER_HISTORY_MESSAGE_COUNT,
        )

        if (
            not user_messages
        ):  # USER_HISTORY_MESSAGE_COUNT (e.g. 200) might be too high to always have messages.
            await processing_message.edit_text(
                "I don't have enough of your messages in this chat to generate a profile yet (need some from your recent history). Keep chatting!"
            )
            return

        # Format chat history for Gemini
        formatted_history = "Here is the user's recent chat history in this group:\n\n"
        for msg in user_messages:
            timestamp = msg.date.strftime("%Y-%m-%d %H:%M:%S")
            # We already know it's the user's message, so no need to repeat username
            formatted_history += f"{timestamp}: {msg.text}\n"

        system_prompt = PROFILEME_SYSTEM_PROMPT
        MAX_CUSTOM_PROMPT_LENGTH = 80
        custom_prompt = (
            (update.effective_message.text or "").replace("/profileme", "", 1).strip()
        )
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
            use_search_grounding=False,  # Probably not needed for profiling
        )

        if profile_response:
            await send_response(
                processing_message, profile_response, "Your User Profile"
            )
        else:
            await processing_message.edit_text(
                "I couldn't generate a profile at this time. Please try again later."
            )

    except Exception as e:  # noqa: BLE001
        logger.error("Error in profileme_handler: %s", e, exc_info=True)
        try:
            await processing_message.edit_text(
                f"Sorry, an error occurred while generating your profile: {str(e)}"
            )
        except Exception:  # noqa: BLE001 # Inner exception during error reporting
            pass


async def help_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:  # noqa: ARG001
    """Handle the /help command.

    Args:
        update: The update containing the message.
        context: The context object.
    """
    if update.effective_message is None or update.effective_chat is None:
        return

    # Check access control
    if not await check_access_control(update, "help"):
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

    /support - Show support information and Ko-fi link
    Usage: `/support`

    /help - Show this help message
    """

    await update.effective_message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)


async def support_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:  # noqa: ARG001
    """Handle the /support command.

    Args:
        update: The update containing the message.
        context: The context object.
    """
    if update.effective_message is None or update.effective_chat is None:
        return

    # Check access control
    if not await check_access_control(update, "support"):
        return

    # Create the support message with inline keyboard
    # Create inline keyboard with Ko-fi link button
    keyboard = [[InlineKeyboardButton("Support the bot", url=SUPPORT_LINK)]]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.effective_message.reply_text(
        SUPPORT_MESSAGE, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN
    )
