"""Configuration module for the TelegramGroupHelperBot."""

import os
from typing import Final, Literal

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Bot configuration
BOT_TOKEN: Final[str] = os.getenv("BOT_TOKEN", "")
if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN environment variable is not set")

GEMINI_API_KEY: Final[str] = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

# Database configuration
DATABASE_URL: Final[str] = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///bot.db")

# Webhook configuration
ENV: Final[Literal["development", "production"]] = os.getenv("ENV", "development") == "production" and "production" or "development"
WEBHOOK_URL: Final[str] = os.getenv("WEBHOOK_URL", "")
WEBHOOK_PORT: Final[int] = int(os.getenv("WEBHOOK_PORT", "8080"))

# Rate limiting (seconds)
RATE_LIMIT: Final[int] = 3

# System prompts
TLDR_PROMPT: Final[str] = "你是一个中文助手，请用中文总结以下群聊内容，限制在 200 字以内。"
FACTCHECK_PROMPT: Final[str] = """
You are a multilingual fact-checker.
Cross-check the following statement against reliable web sources.
Respond in the *same language* as the statement.
Cite sources inline as numbered Markdown links.
"""

# Model configuration
GEMINI_MODEL: Final[str] = "gemini-2.5-flash-preview-04-17"
GEMINI_TEMPERATURE: Final[float] = 0.4
GEMINI_TOP_P: Final[float] = 0.95
GEMINI_TOP_K: Final[int] = 64
GEMINI_MAX_OUTPUT_TOKENS: Final[int] = 65536 