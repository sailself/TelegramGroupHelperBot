import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from google.genai import types as genai_types # For mocking response structure

# Assuming llm.py is in bot.llm
from bot.llm import generate_video_with_veo, GEMINI_VIDEO_MODEL, _safety_settings, GEMINI_MAX_OUTPUT_TOKENS, GEMINI_TEMPERATURE, GEMINI_TOP_P, GEMINI_TOP_K 

class TestGenerateVideoWithVeo(unittest.IsolatedAsyncioTestCase):

    @patch('bot.llm.genai.Client')
    async def test_generate_video_success_with_image(self, MockGeminiClient):
        mock_client_instance = MockGeminiClient.return_value
        # mock_generate_content = AsyncMock() # For asyncio.to_thread
        
        # Mock the response structure from Gemini
        mock_response = MagicMock()
        mock_candidate = MagicMock()
        mock_part = MagicMock()
        mock_part.inline_data = MagicMock(spec=genai_types.Blob)
        mock_part.inline_data.data = b"mock_video_bytes"
        mock_part.inline_data.mime_type = "video/mp4"
        mock_candidate.content = MagicMock() # Ensure content attribute exists
        mock_candidate.content.parts = [mock_part]
        mock_response.candidates = [mock_candidate]
        
        # Wrap synchronous method for asyncio.to_thread
        sync_generate_content = MagicMock(return_value=mock_response)
        mock_client_instance.models.generate_content = sync_generate_content
        
        prompt = "test video with image"
        image_data = b"fake_image_data"
        
        # Mock detect_mime_type as it's called internally
        with patch('bot.llm.detect_mime_type', return_value="image/jpeg") as mock_detect_mime:
            video_bytes, mime_type = await generate_video_with_veo("system", prompt, image_data=image_data)

        self.assertEqual(video_bytes, b"mock_video_bytes")
        self.assertEqual(mime_type, "video/mp4")
        
        mock_detect_mime.assert_called_once_with(image_data)
        
        args, kwargs = sync_generate_content.call_args
        # Check basic call structure
        self.assertEqual(kwargs['model'], GEMINI_VIDEO_MODEL)
        # Check contents (prompt, image part)
        self.assertEqual(args[0][0], "system") # System prompt
        self.assertEqual(args[0][1], prompt)   # User prompt
        self.assertIsInstance(args[0][2], genai_types.Part) # Image part
        self.assertEqual(args[0][2].inline_data.data, image_data)
        self.assertEqual(args[0][2].inline_data.mime_type, "image/jpeg")
        # Check generation_config and safety_settings
        self.assertIn('generation_config', kwargs)
        self.assertEqual(kwargs['generation_config'].response_modalities, ['VIDEO'])
        self.assertEqual(kwargs['safety_settings'], _safety_settings)


    @patch('bot.llm.genai.Client')
    async def test_generate_video_success_prompt_only(self, MockGeminiClient):
        mock_client_instance = MockGeminiClient.return_value
        sync_generate_content = MagicMock() 
        mock_client_instance.models.generate_content = sync_generate_content

        mock_response = MagicMock()
        mock_candidate = MagicMock()
        mock_part = MagicMock()
        mock_part.inline_data = MagicMock(spec=genai_types.Blob)
        mock_part.inline_data.data = b"prompt_only_video"
        mock_part.inline_data.mime_type = "video/mpeg"
        mock_candidate.content = MagicMock()
        mock_candidate.content.parts = [mock_part]
        mock_response.candidates = [mock_candidate]
        sync_generate_content.return_value = mock_response
        
        prompt = "prompt only test"
        video_bytes, mime_type = await generate_video_with_veo("system", prompt, image_data=None)

        self.assertEqual(video_bytes, b"prompt_only_video")
        self.assertEqual(mime_type, "video/mpeg")
        sync_generate_content.assert_called_once()
        args, kwargs = sync_generate_content.call_args
        self.assertEqual(args[0], ["system", prompt]) # Contents should only be system and user prompt


    @patch('bot.llm.genai.Client')
    async def test_generate_video_no_video_data_in_response(self, MockGeminiClient):
        mock_client_instance = MockGeminiClient.return_value
        sync_generate_content = MagicMock()
        mock_client_instance.models.generate_content = sync_generate_content

        mock_response = MagicMock()
        mock_candidate = MagicMock()
        mock_text_part = MagicMock()
        mock_text_part.text = "This is not a video."
        mock_text_part.inline_data = None 
        mock_candidate.content = MagicMock()
        mock_candidate.content.parts = [mock_text_part]
        mock_response.candidates = [mock_candidate]
        sync_generate_content.return_value = mock_response
        
        video_bytes, mime_type = await generate_video_with_veo("system", "test", image_data=None)

        self.assertIsNone(video_bytes)
        self.assertIsNone(mime_type)

    @patch('bot.llm.genai.Client')
    async def test_generate_video_no_candidates(self, MockGeminiClient):
        mock_client_instance = MockGeminiClient.return_value
        sync_generate_content = MagicMock()
        mock_client_instance.models.generate_content = sync_generate_content
        
        mock_response = MagicMock()
        mock_response.candidates = [] 
        sync_generate_content.return_value = mock_response
        
        video_bytes, mime_type = await generate_video_with_veo("system", "test", image_data=None)

        self.assertIsNone(video_bytes)
        self.assertIsNone(mime_type)

    @patch('bot.llm.genai.Client')
    async def test_generate_video_gemini_api_exception(self, MockGeminiClient):
        mock_client_instance = MockGeminiClient.return_value
        sync_generate_content = MagicMock(side_effect=Exception("Gemini API Error"))
        mock_client_instance.models.generate_content = sync_generate_content
        
        # Patch logger to check for error logging
        with patch('bot.llm.logger.error') as mock_logger_error:
            video_bytes, mime_type = await generate_video_with_veo("system", "test exception", image_data=None)

            self.assertIsNone(video_bytes)
            self.assertIsNone(mime_type)
            mock_logger_error.assert_called_with(
                "Error during VEO video generation or response processing: Gemini API Error", exc_info=True
            )


    @patch('bot.llm.genai.Client')
    @patch('bot.llm.detect_mime_type') 
    async def test_generate_video_invalid_image_mime_type(self, mock_detect_mime_type, MockGeminiClient):
        mock_client_instance = MockGeminiClient.return_value
        sync_generate_content = MagicMock()
        mock_client_instance.models.generate_content = sync_generate_content
        
        mock_detect_mime_type.return_value = "application/pdf" 
        
        prompt = "test invalid image mime"
        image_data = b"fake_pdf_data" 
        
        mock_response = MagicMock()
        mock_candidate = MagicMock()
        mock_part = MagicMock()
        mock_part.inline_data = MagicMock(spec=genai_types.Blob)
        mock_part.inline_data.data = b"video_no_image"
        mock_part.inline_data.mime_type = "video/mp4"
        mock_candidate.content = MagicMock()
        mock_candidate.content.parts = [mock_part]
        mock_response.candidates = [mock_candidate]
        sync_generate_content.return_value = mock_response

        # Patch logger to check for error logging regarding invalid MIME
        with patch('bot.llm.logger.error') as mock_logger_error:
            video_bytes, mime_type = await generate_video_with_veo("system", prompt, image_data=image_data)

            self.assertEqual(video_bytes, b"video_no_image")
            self.assertEqual(mime_type, "video/mp4")
            
            args_call, _ = sync_generate_content.call_args
            # Check that the contents list does NOT include the invalid image part
            self.assertEqual(args_call[0], ["system", prompt]) 
            mock_detect_mime_type.assert_called_once_with(image_data)
            mock_logger_error.assert_any_call(
                 "Invalid MIME type for image_data: application/pdf. Skipping image for video generation."
            )

if __name__ == '__main__':
    unittest.main()
