"""Integration test for the TLDR command."""

import asyncio
import datetime
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import Chat, Message, Update, User
from telegram.ext import CallbackContext, ContextTypes

from bot.config import TLDR_PROMPT
from bot.db.models import Message as DbMessage
from bot.handlers import tldr_handler


class TestTLDRIntegration(unittest.TestCase):
    """Integration test for the TLDR command."""

    @pytest.mark.asyncio
    async def test_tldr_integration(self):
        """Test that TLDR works end-to-end with 10 messages and produces a response ≤200 chars."""
        # Mock the database query
        mock_messages = []
        for i in range(10):
            mock_message = DbMessage(
                id=i,
                chat_id=123456789,
                user_id=i + 100,
                username=f"user{i}",
                text=f"This is message {i}",
                language="en",
                date=datetime.datetime.utcnow(),
                reply_to_message_id=None
            )
            mock_messages.append(mock_message)
        
        # Mock Update and Context objects
        update = MagicMock(spec=Update)
        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        
        # Set up command arguments
        context.args = ["10"]
        
        # Mock message and chat
        chat_mock = MagicMock(spec=Chat)
        chat_mock.id = 123456789
        user_mock = MagicMock(spec=User)
        user_mock.id = 987654321
        
        message_mock = MagicMock(spec=Message)
        message_mock.chat = chat_mock
        message_mock.from_user = user_mock
        message_mock.reply_text.return_value = AsyncMock()
        
        # Mock effective_message
        update.effective_message = message_mock
        
        # Set up patch for the database query
        with patch("bot.handlers.get_last_n_text_messages") as mock_get_messages:
            # Set up the return value for the DB query
            mock_get_messages.return_value = mock_messages
            
            # Set up patch for rate limiting
            with patch("bot.handlers.is_rate_limited", return_value=False):
                # Set up patch for Gemini call
                with patch("bot.handlers.call_gemini") as mock_call_gemini:
                    # Generate a Chinese response that is exactly 200 characters long
                    chinese_response = "这" * 200
                    mock_call_gemini.return_value = chinese_response
                    
                    # Call the handler
                    await tldr_handler(update, context)
                    
                    # Verify the handler called the database function with correct params
                    mock_get_messages.assert_called_once_with(chat_mock.id, 10)
                    
                    # Verify Gemini was called with the correct system prompt
                    mock_call_gemini.assert_called_once()
                    args, kwargs = mock_call_gemini.call_args
                    self.assertEqual(kwargs["system_prompt"], TLDR_PROMPT)
                    
                    # Verify that a response was sent and has ≤200 characters
                    message_mock.reply_text.assert_called()
                    reply_message = message_mock.reply_text.return_value
                    
                    # Verify the edit_text call contains a response with ≤200 chars
                    reply_message.edit_text.assert_called_once()
                    args, _ = reply_message.edit_text.call_args
                    response_text = args[0]
                    self.assertLessEqual(len(response_text), 203)  # 200 + "..."


if __name__ == "__main__":
    unittest.main() 