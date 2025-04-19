"""Main entry point for the TelegramGroupHelperBot."""

import asyncio
import logging
import os
import sys
from typing import NoReturn

from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
)

from bot import BOT_TOKEN, ENV, WEBHOOK_PORT, WEBHOOK_URL
from bot.db import db_writer, init_db
from bot.handlers import factcheck_handler, log_message, q_handler, tldr_handler

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


async def start(application: Application) -> None:
    """Start the bot.
    
    Args:
        application: The application instance.
    """
    # Initialize database
    await init_db()
    
    # Start the database writer
    asyncio.create_task(db_writer())
    
    # Start the bot
    if ENV == "production":
        # Use webhook in production
        await application.bot.set_webhook(url=WEBHOOK_URL)
        # Remove the '/' at the beginning of the URL for the webhook path
        webhook_path = WEBHOOK_URL.split("/")[-1]
        await application.start_webhook(
            listen="0.0.0.0",
            port=WEBHOOK_PORT,
            webhook_path=webhook_path,
            drop_pending_updates=True,
        )
        logger.info(f"Bot started in production mode with webhook: {WEBHOOK_URL}")
    else:
        # Use polling in development
        await application.start_polling(drop_pending_updates=True)
        logger.info("Bot started in development mode with polling")
    
    # Keep the program running
    await application.updater.stop()


async def stop(application: Application) -> None:
    """Stop the bot gracefully.
    
    Args:
        application: The application instance.
    """
    if ENV == "production":
        await application.bot.delete_webhook()
    
    await application.stop()
    logger.info("Bot stopped gracefully")


async def main() -> None:
    """Initialize and start the bot."""
    # Create the Application instance
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Register handlers
    application.add_handler(CommandHandler("tldr", tldr_handler))
    application.add_handler(CommandHandler("factcheck", factcheck_handler))
    application.add_handler(CommandHandler("q", q_handler))
    
    # Add a message handler to log all messages
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, log_message))
    
    # Add a healthcheck endpoint
    if ENV == "production":
        from aiohttp import web
        
        async def healthz(_) -> web.Response:
            """Health check endpoint."""
            return web.Response(text="OK")
        
        application.web_app.router.add_get("/healthz", healthz)
    
    # Set up proper signal handling
    application.run_until_complete(start(application))
    
    try:
        await application.updater.initialize()
        await application.updater.start_polling()
        await application.idle()
    finally:
        await stop(application)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1) 