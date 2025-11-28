"""Timing utilities for command and LLM performance logging."""
from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Awaitable, Callable, Mapping, Optional, TypeVar

from telegram import Update

from bot.config import TIMING_LOGGER

T = TypeVar("T")
Handler = Callable[[Update, Any], Awaitable[Any]]


class CommandTimer:
    """Track timing for a Telegram command lifecycle."""

    def __init__(self, command: str, update: Update, logger: logging.Logger) -> None:
        self.command = command
        self.update = update
        self.logger = logger
        self.received_at = datetime.now(timezone.utc)
        self.status = "success"
        self.detail: Optional[str] = None
        self._completed = False

    def _base_metadata(self) -> dict[str, Any]:
        message = self.update.effective_message
        chat = self.update.effective_chat
        user = message.from_user if message else None
        text = (message.text or message.caption or "") if message else ""
        text = text.replace("\n", " ")
        if len(text) > 300:
            text = text[:300]
        return {
            "chat_id": chat.id if chat else None,
            "user_id": user.id if user else None,
            "username": user.username if user else None,
            "message_id": message.message_id if message else None,
            "text": text,
        }

    def log_received(self) -> None:
        metadata = self._base_metadata()
        self.logger.info(
            (
                "event=command_received command=%s chat_id=%s user_id=%s "
                "username=%s message_id=%s received_at=%s text=%s"
            ),
            self.command,
            metadata["chat_id"],
            metadata["user_id"],
            metadata["username"],
            metadata["message_id"],
            self.received_at.isoformat(),
            metadata["text"],
        )

    def mark_status(self, status: str, *, detail: Optional[str] = None) -> None:
        self.status = status
        if detail is not None:
            self.detail = detail

    def log_completed(self) -> None:
        if self._completed:
            return
        self._completed = True
        completed_at = datetime.now(timezone.utc)
        duration = (completed_at - self.received_at).total_seconds()
        metadata = self._base_metadata()
        self.logger.info(
            (
                "event=command_completed command=%s chat_id=%s user_id=%s "
                "message_id=%s started_at=%s response_sent_at=%s "
                "duration_s=%.3f status=%s detail=%s"
            ),
            self.command,
            metadata["chat_id"],
            metadata["user_id"],
            metadata["message_id"],
            self.received_at.isoformat(),
            completed_at.isoformat(),
            duration,
            self.status,
            self.detail or "",
        )


@asynccontextmanager
async def command_timing(command: str, update: Update) -> AsyncIterator[CommandTimer]:
    """Async context manager to log command start/end times."""

    timer = CommandTimer(command=command, update=update, logger=TIMING_LOGGER)
    timer.log_received()
    try:
        yield timer
    except Exception:
        timer.mark_status("error")
        raise
    finally:
        timer.log_completed()


def start_command_timer(command: str, update: Update) -> CommandTimer:
    """Start a command timer and log the receipt."""

    timer = CommandTimer(command=command, update=update, logger=TIMING_LOGGER)
    timer.log_received()
    return timer


def complete_command_timer(
    timer: CommandTimer | None, *, status: str = "success", detail: str | None = None
) -> None:
    """Complete a command timer if provided."""

    if timer is None:
        return
    timer.mark_status(status, detail=detail)
    timer.log_completed()


def wrap_with_command_timing(command: str, handler: Handler) -> Handler:
    """Wrap a handler to emit timing logs for the command lifecycle."""

    async def wrapper(update: Update, context: Any) -> Any:
        if update.effective_message is None or update.effective_chat is None:
            return None
        async with command_timing(command, update):
            return await handler(update, context)

    return wrapper


async def log_llm_timing(
    provider: str,
    model: str,
    operation: str,
    call: Callable[[], Awaitable[T]],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> T:
    """Measure and log timing for an LLM request/response."""

    started_at = datetime.now(timezone.utc)
    started_perf = time.perf_counter()
    metadata_text = json.dumps(metadata or {}, ensure_ascii=True, default=str)
    status = "success"
    TIMING_LOGGER.info(
        "event=llm_request provider=%s model=%s operation=%s started_at=%s metadata=%s",
        provider,
        model,
        operation,
        started_at.isoformat(),
        metadata_text,
    )
    try:
        result = await call()
        return result
    except Exception:
        status = "error"
        raise
    finally:
        completed_at = datetime.now(timezone.utc)
        duration = time.perf_counter() - started_perf
        TIMING_LOGGER.info(
            (
                "event=llm_response provider=%s model=%s operation=%s "
                "completed_at=%s duration_s=%.3f status=%s metadata=%s"
            ),
            provider,
            model,
            operation,
            completed_at.isoformat(),
            duration,
            status,
            metadata_text,
        )
