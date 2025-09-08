"""Test that the factcheck command uses the language of the input message."""

import unittest
from unittest.mock import AsyncMock, patch

import langdetect
import pytest

from bot.config import FACTCHECK_SYSTEM_PROMPT
from bot.llm import call_gemini


class TestFactcheckLanguage(unittest.TestCase):
    """Test factcheck language detection behavior."""

    @pytest.mark.asyncio
    async def test_factcheck_maintains_input_language(self):
        """Test that factcheck responds in the same language as the input."""
        # Test messages in different languages
        test_cases = [
            {"input": "The Earth is flat.", "expected_lang": "en"},
            {"input": "La Terre est plate.", "expected_lang": "fr"},
            {"input": "地球是平的。", "expected_lang": "zh-cn"},
            {"input": "Die Erde ist flach.", "expected_lang": "de"},
        ]
        
        for case in test_cases:
            input_text = case["input"]
            expected_lang = case["expected_lang"]
            
            # Verify the language detection
            detected_lang = langdetect.detect(input_text)
            self.assertEqual(detected_lang[:2], expected_lang[:2], 
                             f"Language detection failed for: {input_text}")
            
            # Mock the Gemini call
            with patch("bot.llm._MODEL.generate_content") as mock_generate:
                # Setup the mock
                mock_response = AsyncMock()
                mock_response.text = f"Response in the same language as: {input_text}"
                mock_generate.return_value = mock_response
                
                # Call the function
                await call_gemini(
                    system_prompt=FACTCHECK_SYSTEM_PROMPT,
                    user_content=input_text,
                    # No response_language
                )
                
                # Assert that no language was specified explicitly
                args, _ = mock_generate.call_args
                messages = args[0]
                
                # There should only be 2 messages (system + user), no language override
                self.assertEqual(len(messages), 2, 
                                f"Expected 2 messages, got {len(messages)}")
                
                # Check that the system prompt is correctly set
                self.assertEqual(messages[0]["role"], "system")
                self.assertEqual(messages[0]["parts"][0], FACTCHECK_SYSTEM_PROMPT)
                
                # Check that the second message contains the user input
                self.assertEqual(messages[1]["role"], "user")
                self.assertEqual(messages[1]["parts"][0], input_text)


if __name__ == "__main__":
    unittest.main() 