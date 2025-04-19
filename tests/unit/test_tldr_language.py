"""Test that TLDR always outputs in Chinese regardless of input language."""

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

import pytest

from bot.config import TLDR_PROMPT
from bot.llm import call_gemini


class TestTLDRLanguage(unittest.TestCase):
    """Test TLDR language behavior."""

    @pytest.mark.asyncio
    async def test_tldr_outputs_chinese(self):
        """Test that TLDR always outputs in Chinese, regardless of input language."""
        # Mock messages in English
        english_messages = "User1: Hello\nUser2: How are you?\nUser1: I'm fine, thanks!"
        
        # Mock the Gemini call
        with patch("bot.llm._MODEL.generate_content") as mock_generate:
            # Setup the mock to return a Chinese response
            mock_response = AsyncMock()
            mock_response.text = "你好！这是一个简短的中文总结。用户之间互相问候。"
            mock_generate.return_value = mock_response
            
            # Call the function with English input
            response = await call_gemini(
                system_prompt=TLDR_PROMPT,
                user_content=english_messages,
                # No response_language here, which is correct behavior
            )
            
            # Assert that the system prompt is the Chinese one from config
            args, _ = mock_generate.call_args
            messages = args[0]
            
            # Check that the system prompt is the Chinese one
            self.assertEqual(messages[0]["role"], "system")
            self.assertEqual(messages[0]["parts"][0], TLDR_PROMPT)
            
            # Check that there's no third message forcing a language
            self.assertEqual(len(messages), 2)
            
            # Check that the response is in Chinese
            self.assertTrue(any(c >= '\u4e00' and c <= '\u9fff' for c in response))


if __name__ == "__main__":
    unittest.main() 