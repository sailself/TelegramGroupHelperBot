"""Configuration module for the TelegramGroupHelperBot."""

import logging
import os
from typing import Dict, List, Optional, cast

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

# Bot settings
# Bot token from BotFather
BOT_TOKEN = os.getenv("BOT_TOKEN")

# Database settings
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///bot.db")

# Gemini settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-pro")
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.7"))
GEMINI_TOP_K = int(os.getenv("GEMINI_TOP_K", "40"))
GEMINI_TOP_P = float(os.getenv("GEMINI_TOP_P", "0.95"))
GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "2048"))

# Whether to use webhooks
USE_WEBHOOK = os.getenv("USE_WEBHOOK", "false").lower() == "true"
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")
WEBHOOK_PORT = int(os.getenv("WEBHOOK_PORT", "8443"))

# Telegram rate limiting (seconds between requests)
RATE_LIMIT_SECONDS = int(os.getenv("RATE_LIMIT_SECONDS", "15"))

# Prompt for a short summary
TLDR_SYSTEM_PROMPT = "你是一个中文助手，请用中文总结以下群聊内容，请列出主要发言用户的名字和观点。限制在 400 字以内。"

# Prompt for fact checking
FACTCHECK_SYSTEM_PROMPT = """You are an expert fact-checker that is unbiased, honest, and direct. \
Your job is to evaluate the factual accuracy of the text provided.

For each significant claim, verify using web search results:
1. Analyze each claim objectively
2. Provide a judgment on its accuracy (True, False, Partially True, or Insufficient Evidence)
3. Briefly explain your reasoning with citations to the sources found through web search
4. When a claim is not factually accurate, provide corrections
5. IMPORTANT: The current date is {current_date}. Verify all temporal claims relative to this date.

Always cite your sources and only draw definitive conclusions when you have sufficient reliable evidence.
"""

# Prompt for the Q&A bot
Q_SYSTEM_PROMPT = """You are a helpful assistant in a Telegram group chat. You provide concise, factual, and helpful answers to users' questions.

Guidelines for your responses:
1. Provide a direct, clear answer to the question.
2. Be concise but comprehensive.
3. Fact-check your information using web search and include citations to reliable sources.
4. When the question asks for technical information, provide accurate and up-to-date information.
5. IMPORTANT: Use web search to verify all facts and information before answering.
6. CRITICAL: The current date is {current_date}. Always verify current political leadership, office holders, and recent events through web search based on this date.
7. If there's uncertainty, acknowledge it and explain the limitations.
8. If the query contains inappropriate content, politely decline to engage.
9. Format your response in an easily readable way using Markdown where appropriate.
10. Keep your response under 400 words unless a detailed explanation is necessary.
11. If the answer requires multiple parts, use numbered or bulleted lists.
12. If the question is in a language other than English, respond in the same language.

Remember to be helpful, accurate, and respectful in your responses.
""" 