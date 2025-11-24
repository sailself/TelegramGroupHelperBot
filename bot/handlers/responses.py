"""Response helpers for TelegramGroupHelperBot."""
from __future__ import annotations

import logging

import langid
from pycountry import languages
from telegram import Update
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from bot.config import TELEGRAM_MAX_LENGTH
from bot.db.database import queue_message_insert

from .content import create_telegraph_page

logger = logging.getLogger(__name__)


async def send_response(
    message, response, title="Response", parse_mode=ParseMode.MARKDOWN
):
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
    line_count = response.count("\n") + 1

    # Check if message exceeds line count threshold or character limit
    if line_count > 22 or len(response) > TELEGRAM_MAX_LENGTH:
        logger.info(
            "Response length %d exceeds threshold %d, creating Telegraph page",
            len(response),
            TELEGRAM_MAX_LENGTH,
        )
        telegraph_url = await create_telegraph_page(title, response)
        if telegraph_url:
            await message.edit_text(
                f"I have too much to say. [View it here]({telegraph_url})", 
                parse_mode=ParseMode.MARKDOWN
            )
        else:
            # Fallback: try to send as plain text
            try:
                await message.edit_text(response)
            except BadRequest:
                # If still too long, truncate
                await message.edit_text(
                    f"{response[:TELEGRAM_MAX_LENGTH - 100]}...\n\n(Response was truncated due to length)"
                )
    else:
        # Message is within limits, try to send with formatting
        try:
            await message.edit_text(response, parse_mode=parse_mode)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to send response with formatting: %s", e)
            # If formatting fails, try to send as plain text
            try:
                await message.edit_text(response)
            except BadRequest as plain_e:
                if "Message_too_long" in str(plain_e):
                    # Handle the case where the message is unexpectedly too long
                    telegraph_url = await create_telegraph_page(title, response)
                    if telegraph_url:
                        await message.edit_text(
                            f"The response is too long. [View it here]({telegraph_url})", 
                            parse_mode=ParseMode.MARKDOWN
                        )
                    else:
                        # Last resort: truncate
                        await message.edit_text(
                            f"{response[:TELEGRAM_MAX_LENGTH - 100]}...\n\n(Response was truncated due to length)"
                        )
                else:
                    logger.error(
                        "Failed to send response as plain text: %s",
                        plain_e,
                        exc_info=True,
                    )
                    await message.edit_text(
                        "Error: Failed to format response. Please try again."
                    )


async def log_message(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:  # noqa: ARG001
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
    language = languages.get(alpha_2=langid.classify(text)[0]).name

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
        reply_to_message_id=(
            message.reply_to_message.message_id if message.reply_to_message else None
        ),
        chat_id=update.effective_chat.id if update.effective_chat else None,
        message_id=message.message_id,
    )
