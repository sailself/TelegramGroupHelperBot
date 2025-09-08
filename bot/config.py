"""Configuration module for the TelegramGroupHelperBot."""

import logging
import os
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Set up logging
# Configure root logger
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
    level=logging.INFO
)

# Get the root logger
root_logger = logging.getLogger()

# Remove any existing handlers
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Create a file handler that rotates daily
file_handler = TimedRotatingFileHandler(
    filename=logs_dir / "bot.log",
    when="midnight",
    interval=1,
    backupCount=30,  # Keep logs for 30 days
    encoding="utf-8"
)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
file_handler.setLevel(logging.INFO)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
console_handler.setLevel(logging.INFO)

# Add handlers to root logger
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

# Create a logger for this module
logger = logging.getLogger(__name__)
logger.info("Logging configured with daily rotation")

# Bot settings
# Bot token from BotFather
BOT_TOKEN = os.getenv("BOT_TOKEN")

# Database settings
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///bot.db")

# Gemini settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_PRO_MODEL = os.getenv("GEMINI_PRO_MODEL", "gemini-2.5-pro-exp-03-25")
GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.0-flash-exp-image-generation")
GEMINI_VIDEO_MODEL = os.getenv("GEMINI_VIDEO_MODEL", "veo-2.0-generate-001")
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.7"))
GEMINI_TOP_K = int(os.getenv("GEMINI_TOP_K", "40"))
GEMINI_TOP_P = float(os.getenv("GEMINI_TOP_P", "0.95"))
GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "2048"))

VERTEX_PROJECT_ID=os.getenv("VERTEX_PROJECT_ID")
VERTEX_LOCATION=os.getenv("VERTEX_LOCATION")
USE_VERTEX_VIDEO=os.getenv("USE_VERTEX_VIDEO", "false").lower() == "true"
VERTEX_VIDEO_MODEL=os.getenv("VERTEX_VIDEO_MODEL")
USE_VERTEX_IMAGE=os.getenv("USE_VERTEX_IMAGE", "false").lower() == "true"
VERTEX_IMAGE_MODEL=os.getenv("VERTEX_IMAGE_MODEL")

# Whether to use webhooks
USE_WEBHOOK = os.getenv("USE_WEBHOOK", "false").lower() == "true"
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")
WEBHOOK_PORT = int(os.getenv("WEBHOOK_PORT", "8443"))

# Telegram rate limiting (seconds between requests)
RATE_LIMIT_SECONDS = int(os.getenv("RATE_LIMIT_SECONDS", "15"))

# Telegram message length threshold (characters) before creating Telegraph page
TELEGRAM_MAX_LENGTH = int(os.getenv("TELEGRAM_MAX_LENGTH", "4000"))

# Telegraph access token
TELEGRAPH_ACCESS_TOKEN = os.getenv("TELEGRAPH_ACCESS_TOKEN", "")
TELEGRAPH_AUTHOR_NAME = os.getenv("TELEGRAPH_AUTHOR_NAME", "")
TELEGRAPH_AUTHOR_URL = os.getenv("TELEGRAPH_AUTHOR_URL", "")

# Number of messages to fetch for user-specific history features
USER_HISTORY_MESSAGE_COUNT = int(os.getenv("USER_HISTORY_MESSAGE_COUNT", "200"))

# CWD.PW image upload settings
CWD_PW_API_KEY = os.getenv("CWD_PW_API_KEY", "")

# Support/Ko-fi settings
SUPPORT_MESSAGE = os.getenv("SUPPORT_MESSAGE", "如果觉得{bot_name}有帮助，可以支持一下它的开发！")
SUPPORT_LINK = os.getenv("SUPPORT_LINK", "")

# Whitelist settings
WHITELIST_FILE_PATH = os.getenv("WHITELIST_FILE_PATH", "allowed_chat.txt")

# Access control settings
ACCESS_CONTROLLED_COMMANDS = os.getenv("ACCESS_CONTROLLED_COMMANDS", "").split(",") if os.getenv("ACCESS_CONTROLLED_COMMANDS") else []

# Prompt for a short summary
TLDR_SYSTEM_PROMPT = """你是一个AI助手，名叫{bot_name}，请用中文总结以下群聊内容。
请先汇总出群聊主要内容。
再依据发言数量依次列出主要发言用户的名字和观点但不要超过10位用户。
请尽量详细地表述每个人的对各个议题的观点和陈述，字数不限。
非常关键：如果群聊内容中出现投资相关信息，请在总结后再全文最后逐项列出。格式为：投资标的物：投资建议 [由哪位用户提出]。
"""
# Prompt for fact checking
FACTCHECK_SYSTEM_PROMPT = """You are an expert fact-checker that is unbiased, honest, and direct. \
Your job is to evaluate the factual accuracy of the text provided.

For each significant claim, verify using web search results:
1. Analyze each claim objectively
2. Provide a judgment on its accuracy (True, False, Partially True, or Insufficient Evidence)
3. Briefly explain your reasoning with citations to the sources found through web search
4. When a claim is not factually accurate, provide corrections
5. IMPORTANT: The current UTC date and time is {current_datetime}. Verify all temporal claims relative to this date and time.
6. CRITICAL: List the sources you used to check the facts with links.
7. CRITICAL: Always respond in the same language as the user's message or the language from the image.
8. Format your response in an easily readable way using Markdown where appropriate.

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
6. CRITICAL: The current UTC date and time is {current_datetime}. Always verify current political leadership, office holders, and recent events through web search based on this date and time.
7. If there's uncertainty, acknowledge it and explain the limitations.
8. Format your response in an easily readable way using Markdown where appropriate.
9. Keep your response under 400 words unless a detailed explanation is necessary.
10. If the answer requires multiple parts, use numbered or bulleted lists.
11. CRITICAL: Respond in {language} language unless you are told otherwise.

Remember to be helpful, accurate, and respectful in your responses.
"""

# Prompt for the /profileme handler
PROFILEME_SYSTEM_PROMPT = (
    "You are an experienced professional profiler. Based on the following chat history of a user in a group chat, "
    "generate a concise and insightful user profile. The profile must highlight their "
    "communication style, potential interests, key personality traits, and how they typically interact in the group. "
    "Focus on patterns and recurring themes. Address the user directly (e.g., 'You seem to be...')."
    "Do not include any specific message content, timestamps or message IDs."
    "The user is asking for their own profile."
    "CRITICAL: Always reply in Chinese"
)

# Prompt for the /paintme handler (image prompt generation)
PAINTME_SYSTEM_PROMPT = (
    "You are a highly creative AI specializing in crafting evocative and detailed image prompts. "
    "Based on the following chat history of a user, generate a concise, yet descriptive and imaginative image prompt "
    "that metaphorically represents the user's personality, communication style, or recurring themes found in their messages. "
    "This prompt will be used by an AI image generation model to create a visual representation of the user. "
    "Focus on symbolism, artistic interpretation, and sensory details. Do not refer to the user directly or mention the chat itself in the prompt. "
    "The final prompt should be a descriptive phrase or a short paragraph. "
    "Strive for prompts that can lead to realistic, artistic, or conceptually rich images. \n\n"
    "The prompt is used for Imagen 4 model, so the prompt should be in English."
)

PORTRAIT_SYSTEM_PROMPT = (
    "You are a highly creative AI specializing in crafting evocative and detailed image prompts for artistic portraits. "
    "Based on the following chat history of a user, generate a concise, yet descriptive and imaginative image prompt "
    "that creates a portrait of the user. The prompt should translate the user's personality, communication style, or recurring themes "
    "into the artistic style, mood, lighting, and background of the portrait. This prompt will be used by an AI image generation model "
    "to create a visual representation of the user themselves. Focus on artistic interpretation and sensory details to produce a compelling image. "
    "Do not refer to the user by a name or mention the chat itself in the prompt. The final prompt should be a descriptive phrase or a short paragraph."
    "The portrait should be a realistic portrait of the user, not a cartoon or a stylized portrait."
    "The prompt is used for Imagen 4 model, so the prompt should be in English."
)