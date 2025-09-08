"""Integration tests for the Telegraph functionality."""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import Bot, Chat, Message, Update, User
from telegram.ext import ContextTypes

from bot.handlers import factcheck_handler, q_handler, send_response


class TestTelegraphIntegration(unittest.TestCase):
    """Integration tests for Telegraph functionality."""

    @pytest.fixture
    def mock_update(self):
        """Create a mock Update object."""
        update = MagicMock(spec=Update)
        update.effective_message = MagicMock(spec=Message)
        update.effective_message.message_id = 123
        update.effective_message.from_user = MagicMock(spec=User)
        update.effective_message.from_user.id = 456
        update.effective_message.from_user.full_name = "Test User"
        update.effective_message.reply_to_message = None
        update.effective_message.date = None
        update.effective_message.text = None
        update.effective_message.reply_text.return_value = MagicMock()
        update.effective_chat = MagicMock(spec=Chat)
        update.effective_chat.id = 789
        update.effective_sender = update.effective_message.from_user
        return update

    @pytest.fixture
    def mock_context(self):
        """Create a mock Context object."""
        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.bot = MagicMock(spec=Bot)
        context.args = []
        return context

    @pytest.mark.asyncio
    async def test_send_response_with_long_message(self, mock_update):
        """Test sending a response that exceeds the character limit."""
        # Mock message
        message = MagicMock()
        message.edit_text = MagicMock()
        
        # Long response (exceeds default threshold)
        long_response = "A" * 5000
        
        # Mock create_telegraph_page function
        with patch('bot.handlers.create_telegraph_page', new_callable=AsyncMock) as mock_telegraph:
            # Configure the mock to return a telegraph URL
            mock_telegraph.return_value = "https://telegra.ph/test-page"
            
            # Call the send_response function
            await send_response(message, long_response, "Test Title")
            
            # Verify create_telegraph_page was called
            mock_telegraph.assert_called_once()
            args, _ = mock_telegraph.call_args
            self.assertEqual(args[0], "Test Title")
            self.assertEqual(args[1], long_response)
            
            # Verify message.edit_text was called with the telegraph URL
            message.edit_text.assert_called_once()
            args, _ = message.edit_text.call_args
            self.assertIn("https://telegra.ph/test-page", args[0])

    @pytest.mark.asyncio
    async def test_q_handler_with_long_response(self, mock_update, mock_context):
        """Test q_handler with a response that would exceed character limit."""
        # Set up the query
        mock_update.effective_message.text = "/q What is the meaning of life?"
        mock_context.args = ["What", "is", "the", "meaning", "of", "life?"]
        
        # Mock the processing message
        processing_message = MagicMock()
        mock_update.effective_message.reply_text.return_value = processing_message
        
        # Mock call_gemini to return a long response
        with patch('bot.handlers.call_gemini', new_callable=AsyncMock) as mock_gemini:
            long_response = "A" * 5000
            mock_gemini.return_value = long_response
            
            # Mock send_response function
            with patch('bot.handlers.send_response', new_callable=AsyncMock) as mock_send:
                # Run the handler
                await q_handler(mock_update, mock_context)
                
                # Verify call_gemini was called
                mock_gemini.assert_called_once()
                
                # Verify send_response was called with the long response
                mock_send.assert_called_once()
                args, kwargs = mock_send.call_args
                self.assertEqual(args[0], processing_message)
                self.assertEqual(args[1], long_response)

    @pytest.mark.asyncio
    async def test_factcheck_handler_with_long_response(self, mock_update, mock_context):
        """Test factcheck_handler with a response that would exceed character limit."""
        # Set up reply to message
        mock_update.effective_message.reply_to_message = MagicMock(spec=Message)
        mock_update.effective_message.reply_to_message.text = "The Earth is flat."
        mock_update.effective_message.reply_to_message.message_id = 100
        
        # Mock the processing message
        processing_message = MagicMock()
        mock_update.effective_message.reply_text.return_value = processing_message
        
        # Mock stream_gemini to add long content to the queue
        with patch('bot.handlers.stream_gemini', new_callable=AsyncMock) as mock_gemini:
            # Create a mock queue
            mock_queue = MagicMock()
            
            # Set up the queue to return a long response followed by None
            mock_queue.get.side_effect = ["A" * 5000, None]
            mock_gemini.return_value = mock_queue
            
            # Mock send_response function
            with patch('bot.handlers.send_response', new_callable=AsyncMock) as mock_send:
                # Run the handler
                await factcheck_handler(mock_update, mock_context)
                
                # Verify stream_gemini was called
                mock_gemini.assert_called_once()
                
                # Verify send_response was called with the long response
                mock_send.assert_called_once()
                args, kwargs = mock_send.call_args
                self.assertEqual(args[0], processing_message)
                self.assertEqual(args[1], "A" * 5000) 