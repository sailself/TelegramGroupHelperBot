"""Main entry point for the TelegramGroupHelperBot."""

import asyncio
import logging
import sys

from telegram.ext import (
    Application,
    CallbackQueryHandler,
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
    handle_media_group,
    help_handler,
    img_handler,
    load_whitelist,
    log_message,
    model_selection_callback,
    paintme_handler,
    profileme_handler,
    deepseek_handler,
    qwen_handler,
    llama_handler,
    gpt_handler,
    q_handler,
    start_handler,
    start_periodic_cleanup,
    stop_periodic_cleanup,
    support_handler,
    tldr_handler,
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
    """Initialize the database and load whitelist."""
    await init_db()
    logger.info("Database initialization completed")
    
    # Load whitelist cache
    load_whitelist()
    logger.info("Whitelist cache loaded")


async def post_init(application):
    """Post-initialization tasks."""
    # Start the periodic cleanup task
    await start_periodic_cleanup(application.bot)
    logger.info("Post-initialization completed")


def main():
    """Main function to run the bot."""
    # Create the Application
    application = Application.builder().token(BOT_TOKEN).concurrent_updates(True).build()
    
    # Register handlers
    application.add_handler(CommandHandler("start", start_handler))
    application.add_handler(CommandHandler("help", help_handler))
    application.add_handler(CommandHandler("tldr", tldr_handler))
    application.add_handler(CommandHandler("factcheck", factcheck_handler))
    application.add_handler(CommandHandler("q", q_handler))
    application.add_handler(CommandHandler("deepseek", deepseek_handler))
    application.add_handler(CommandHandler("qwen", qwen_handler))
    application.add_handler(CommandHandler("llama", llama_handler))
    application.add_handler(CommandHandler("gpt", gpt_handler))
    application.add_handler(CommandHandler("img", img_handler))
    application.add_handler(CommandHandler("vid", vid_handler))
    application.add_handler(CommandHandler("profileme", profileme_handler))
    application.add_handler(CommandHandler("paintme", paintme_handler))
    application.add_handler(CommandHandler("portraitme", paintme_handler))
    application.add_handler(CommandHandler("support", support_handler))
    
    # Callback query handler for model selection
    application.add_handler(CallbackQueryHandler(model_selection_callback))
    
    # Handler for media groups
    application.add_handler(MessageHandler(filters.PHOTO & ~filters.COMMAND, handle_media_group), group=1)

    # Add a message handler to log all messages
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, log_message))
    
    # Initialize database before starting
    loop = asyncio.get_event_loop()
    loop.run_until_complete(init_db_wrapper())
    
    # Post-initialization tasks
    loop.run_until_complete(post_init(application))
    
    # Configure health check endpoint if using webhook
    if USE_WEBHOOK:        
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
        # Stop the cleanup task
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(stop_periodic_cleanup())
        else:
            loop.run_until_complete(stop_periodic_cleanup())
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        # Stop the cleanup task
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(stop_periodic_cleanup())
        else:
            loop.run_until_complete(stop_periodic_cleanup())
        sys.exit(1) 
