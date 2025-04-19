"""Integration test for the Q command when used as a reply."""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import Chat, Message, Update, User
from telegram.ext import ContextTypes

from bot.handlers import q_handler


class TestQReplyIntegration(unittest.TestCase):
    """Integration test for the Q command when used as a reply."""

    @pytest.mark.asyncio
    async def test_q_reply_integration(self):
        """Test that replying 'Who wrote Hamlet?' with /q includes the context and produces an answer."""
        # Mock Update and Context objects
        update = MagicMock(spec=Update)
        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        
        # Set up command arguments (empty for a pure reply scenario)
        context.args = ["Who", "wrote", "Hamlet?"]
        
        # Mock user, chat, message and reply
        chat_mock = MagicMock(spec=Chat)
        chat_mock.id = 123456789
        
        user_mock = MagicMock(spec=User)
        user_mock.id = 987654321
        
        # Create the replied-to message
        replied_message = MagicMock(spec=Message)
        replied_message.text = "I'm wondering about the authorship of a famous play."
        
        # Create the command message
        message_mock = MagicMock(spec=Message)
        message_mock.chat = chat_mock
        message_mock.from_user = user_mock
        message_mock.reply_to_message = replied_message
        message_mock.reply_text.return_value = AsyncMock()
        
        # Set effective_message
        update.effective_message = message_mock
        
        # Set up patch for rate limiting
        with patch("bot.handlers.is_rate_limited", return_value=False):
            # Set up patch for language detection
            with patch("langdetect.detect", return_value="en"):
                # Set up patch for Gemini call
                with patch("bot.handlers.call_gemini") as mock_call_gemini:
                    # Set a mock response
                    mock_call_gemini.return_value = "William Shakespeare wrote Hamlet."
                    
                    # Call the handler
                    await q_handler(update, context)
                    
                    # Verify Gemini was called with the replied-to message prepended
                    mock_call_gemini.assert_called_once()
                    args, kwargs = mock_call_gemini.call_args
                    
                    # Check that the content includes both the replied message and the question
                    self.assertIn("> I'm wondering about the authorship of a famous play.", kwargs["user_content"])
                    self.assertIn("Who wrote Hamlet?", kwargs["user_content"])
                    
                    # Verify that a response was sent
                    message_mock.reply_text.assert_called()
                    reply_message = message_mock.reply_text.return_value
                    
                    # Verify the edit_text call contains Shakespeare
                    reply_message.edit_text.assert_called_once()
                    args, _ = reply_message.edit_text.call_args
                    response_text = args[0]
                    self.assertIn("Shakespeare", response_text)


if __name__ == "__main__":
    unittest.main() 