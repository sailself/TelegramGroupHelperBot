"""Main entry point for the TelegramGroupHelperBot."""

import asyncio
import logging
import os
import signal
import sys
from typing import NoReturn

from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
)

from bot.config import (
    BOT_TOKEN,
    USE_WEBHOOK,
    WEBHOOK_PORT,
    WEBHOOK_URL,
)
from bot.db.database import init_db
from bot.handlers import (
    factcheck_handler,
    help_handler,
    log_message,
    q_handler,
    start_handler,
    tldr_handler,
    img_handler,
    vid_handler,
)

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
# Set DEBUG level for bot handlers to trace issues
logging.getLogger('bot.handlers').setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


async def init_db_wrapper():
    """Initialize the database."""
    await init_db()
    logger.info("Database initialization completed")


def main():
    """Main function to run the bot."""
    # Create the Application
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Register handlers
    application.add_handler(CommandHandler("start", start_handler))
    application.add_handler(CommandHandler("help", help_handler))
    application.add_handler(CommandHandler("tldr", tldr_handler))
    application.add_handler(CommandHandler("factcheck", factcheck_handler))
    application.add_handler(CommandHandler("q", q_handler))
    application.add_handler(CommandHandler("img", img_handler))
    application.add_handler(CommandHandler("vid", vid_handler))
    
    # Add a message handler to log all messages
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, log_message))
    
    # Initialize database before starting
    loop = asyncio.get_event_loop()
    loop.run_until_complete(init_db_wrapper())
    
    # Configure health check endpoint if using webhook
    if USE_WEBHOOK:
        from aiohttp import web
        
        async def healthz(_) -> web.Response:
            """Health check endpoint."""
            return web.Response(text="OK")
        
        application.web_app.router.add_get("/healthz", healthz)
        
        # Start bot with webhook
        logger.info(f"Starting bot in production mode with webhook: {WEBHOOK_URL}")
        application.run_webhook(
            listen="0.0.0.0",
            port=WEBHOOK_PORT,
            webhook_url=WEBHOOK_URL,
            drop_pending_updates=True
        )
    else:
        # Start bot with polling
        logger.info("Starting bot in development mode with polling")
        application.run_polling(drop_pending_updates=True)
    
    logger.info("Bot started successfully")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1) 