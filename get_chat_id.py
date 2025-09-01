#!/usr/bin/env python3
"""
Simple script to get Telegram chat IDs.
This script helps you identify user IDs and group chat IDs for the whitelist.

Usage:
1. Set your bot token in the BOT_TOKEN variable below
2. Run this script: python get_chat_id.py
3. Send a message to your bot
4. The script will show you the user ID and chat ID
"""

import asyncio
import logging
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes

# Set your bot token here
BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming messages and display ID information."""
    user = update.effective_user
    chat = update.effective_chat
    
    print(f"\n{'='*50}")
    print(f"Message received!")
    print(f"{'='*50}")
    print(f"User ID: {user.id}")
    print(f"Username: @{user.username}" if user.username else "Username: None")
    print(f"Full Name: {user.full_name}")
    print(f"Chat ID: {chat.id}")
    print(f"Chat Type: {chat.type}")
    print(f"Chat Title: {chat.title}" if chat.title else "Chat Title: None")
    print(f"{'='*50}")
    
    # Send a response back
    response = (
        f"🔍 **ID Information**\n\n"
        f"👤 **User ID**: `{user.id}`\n"
        f"👤 **Username**: @{user.username}\n" if user.username else f"👤 **Username**: None\n"
        f"👤 **Full Name**: {user.full_name}\n"
        f"💬 **Chat ID**: `{chat.id}`\n"
        f"💬 **Chat Type**: {chat.type}\n"
        f"💬 **Chat Title**: {chat.title}\n" if chat.title else f"💬 **Chat Title**: None\n"
        f"\n"
        f"📝 **For Whitelist**:\n"
        f"- Add `{user.id}` to allow this user\n"
        f"- Add `{chat.id}` to allow this chat/group\n"
    )
    
    await update.message.reply_text(response, parse_mode='Markdown')

async def main():
    """Main function to run the bot."""
    if BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("❌ Please set your bot token in the BOT_TOKEN variable!")
        return
    
    print("🤖 Starting chat ID helper bot...")
    print("📱 Send a message to your bot to get ID information")
    print("⏹️  Press Ctrl+C to stop the bot")
    
    # Create the Application
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Add message handler
    application.add_handler(MessageHandler(filters.ALL, handle_message))
    
    # Start the bot
    await application.run_polling()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
