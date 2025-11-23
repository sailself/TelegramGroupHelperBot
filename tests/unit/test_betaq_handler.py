import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from telegram import Chat, Message, Update
from telegram.ext import ContextTypes

from bot.handlers.qa import betaq_handler


class TestBetaQHandler(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.update = MagicMock(spec=Update)
        self.context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)

        self.update.effective_chat = MagicMock(spec=Chat)
        self.update.effective_chat.id = -100987654321
        self.update.effective_message = MagicMock(spec=Message)
        self.update.effective_message.reply_text = AsyncMock()

    @patch("bot.handlers.qa.ENABLE_PINECONE_RAG", True)
    @patch("bot.handlers.qa.retrieve_formatted_context", new_callable=AsyncMock)
    @patch("bot.handlers.qa.q_handler", new_callable=AsyncMock)
    async def test_betaq_routes_through_q_handler_with_rag(
        self, mock_q_handler, mock_retrieve
    ):
        mock_retrieve.return_value = "ctx-snippet"

        await betaq_handler(self.update, self.context)

        mock_q_handler.assert_awaited_once()
        _, kwargs = mock_q_handler.await_args
        self.assertEqual(kwargs.get("model_name"), "Gemini RAG")

        call_model = kwargs.get("call_model")
        self.assertIsNotNone(call_model, "betaq_handler should forward a call_model")

        with patch("bot.handlers.qa.call_gemini", new_callable=AsyncMock) as mock_call:
            await call_model(system_prompt="sys", user_content="question")

            mock_call.assert_awaited_once()
            mock_retrieve.assert_awaited_once_with(
                self.update.effective_chat.id, "question"
            )
            _, call_kwargs = mock_call.await_args
            self.assertIn("ctx-snippet", call_kwargs["user_content"])
            self.assertIn("question", call_kwargs["user_content"])
