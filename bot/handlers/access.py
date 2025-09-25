"""Access control and rate limiting helpers for TelegramGroupHelperBot."""
from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional

from telegram import Update

from bot.config import (
    ACCESS_CONTROLLED_COMMANDS,
    RATE_LIMIT_SECONDS,
    WHITELIST_FILE_PATH,
)

logger = logging.getLogger(__name__)

user_rate_limits: Dict[int, float] = {}
_whitelist_cache: Optional[List[str]] = None
_whitelist_loaded = False


def is_rate_limited(user_id: int) -> bool:
    """Check if a user is rate limited.

    Args:
        user_id: The ID of the user to check.

    Returns:
        True if the user is rate limited, False otherwise.
    """
    global user_rate_limits  # noqa: PLW0603
    current_time = time.time()

    if user_id in user_rate_limits:
        last_request_time = user_rate_limits[user_id]
        if current_time - last_request_time < RATE_LIMIT_SECONDS:
            return True

    user_rate_limits[user_id] = current_time
    return False


def load_whitelist() -> None:
    """Load whitelist from file into memory cache.

    This function should be called once during application startup.
    """
    global _whitelist_cache, _whitelist_loaded  # noqa: PLW0603

    try:
        # If whitelist file doesn't exist, allow all users (backward compatibility)
        if not os.path.exists(WHITELIST_FILE_PATH):
            logger.info(
                "Whitelist file {WHITELIST_FILE_PATH} not found, allowing all users"
            )
            _whitelist_cache = None
            _whitelist_loaded = True
            return

        with open(WHITELIST_FILE_PATH, "r", encoding="utf-8") as f:
            allowed_ids = [
                line.strip()
                for line in f.readlines()
                if line.strip() and not line.strip().startswith("#")
            ]

        _whitelist_cache = allowed_ids
        _whitelist_loaded = True
        logger.info(
            "Loaded %d entries from whitelist file %s",
            len(allowed_ids),
            WHITELIST_FILE_PATH,
        )

    except Exception as e:  # noqa: BLE001
        logger.error("Error loading whitelist: %s", e)
        # In case of error, deny access for security by setting empty list
        _whitelist_cache = []
        _whitelist_loaded = True


def is_user_whitelisted(user_id: int) -> bool:
    """Check if a user is whitelisted to use the bot.

    Args:
        user_id: The ID of the user to check.

    Returns:
        True if the user is whitelisted, False otherwise.
    """
    global _whitelist_cache, _whitelist_loaded  # noqa: PLW0602

    # If whitelist hasn't been loaded yet, load it
    if not _whitelist_loaded:
        load_whitelist()

    # If whitelist cache is None, it means the file doesn't exist - allow all users
    if _whitelist_cache is None:
        return True

    # Check if user_id is in the cached whitelist
    return str(user_id) in _whitelist_cache


def is_chat_whitelisted(chat_id: int) -> bool:
    """Check if a chat (group/channel) is whitelisted to use the bot.

    Args:
        chat_id: The ID of the chat to check.

    Returns:
        True if the chat is whitelisted, False otherwise.
    """
    global _whitelist_cache, _whitelist_loaded  # noqa: PLW0602

    # If whitelist hasn't been loaded yet, load it
    if not _whitelist_loaded:
        load_whitelist()

    # If whitelist cache is None, it means the file doesn't exist - allow all chats
    if _whitelist_cache is None:
        return True

    # Check if chat_id is in the cached whitelist
    return str(chat_id) in _whitelist_cache


def is_access_allowed(user_id: int, chat_id: int) -> bool:
    """Check if access is allowed for both user and chat.

    Args:
        user_id: The ID of the user to check.
        chat_id: The ID of the chat to check.

    Returns:
        True if access is allowed, False otherwise.
    """
    # Check if user is whitelisted
    user_allowed = is_user_whitelisted(user_id)

    # Check if chat is whitelisted
    chat_allowed = is_chat_whitelisted(chat_id)

    # Access is allowed if either user OR chat is whitelisted
    # This means users can use the bot in whitelisted chats, and whitelisted users can use it anywhere
    return user_allowed or chat_allowed


def requires_access_control(command: str) -> bool:
    """Check if a command requires access control.

    Args:
        command: The command name (without the / prefix)

    Returns:
        True if the command requires access control, False otherwise.
    """
    # If ACCESS_CONTROLLED_COMMANDS is empty or None, apply no access control
    if not ACCESS_CONTROLLED_COMMANDS:
        return False

    # Check if command is in the list of access-controlled commands
    return command in ACCESS_CONTROLLED_COMMANDS


async def check_access_control(update: Update, command: str) -> bool:
    """Check access control for a command if required.

    Args:
        update: The update containing the message.
        command: The command name (without the / prefix)

    Returns:
        True if access is allowed or not required, False if access is denied.

    Side effects:
        Sends an error message to the user if access is denied.
    """
    if update.effective_message is None or update.effective_chat is None:
        return False

    # If command doesn't require access control, allow it
    if not requires_access_control(command):
        return True

    # Check if user has access
    if not is_access_allowed(
        update.effective_message.from_user.id, update.effective_chat.id
    ):
        await update.effective_message.reply_text(
            "You are not authorized to use this command. Please contact the administrator."
        )
        return False

    return True
