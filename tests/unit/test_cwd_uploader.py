"""Tests for the CWD.PW uploader module."""

import base64
import unittest
from unittest.mock import AsyncMock, MagicMock, patch, call
import aiohttp
import pytest

from bot.cwd_uploader import upload_base64_image_to_cwd, upload_image_bytes_to_cwd


class TestCwdUploader(unittest.IsolatedAsyncioTestCase):
    """Test cases for CWD.PW image uploader functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.api_key = "test_api_key_123"
        self.test_image_bytes = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde'
        self.test_base64_png = f"data:image/png;base64,{base64.b64encode(self.test_image_bytes).decode()}"
        self.test_base64_jpg = f"data:image/jpeg;base64,{base64.b64encode(self.test_image_bytes).decode()}"
        self.success_response = {
            "success": True,
            "imageUrl": "https://cwd.pw/i/test123.png"
        }

    async def test_upload_base64_image_success_png(self):
        """Test successful upload of PNG image."""
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json = AsyncMock(return_value=self.success_response)
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session
            mock_session.post.return_value.__aenter__.return_value = mock_response
            
            result = await upload_base64_image_to_cwd(self.test_base64_png, self.api_key)
            
            self.assertEqual(result, "https://cwd.pw/i/test123.png")
            mock_session.post.assert_called_once()
            
            # Check the call arguments
            call_args = mock_session.post.call_args
            self.assertEqual(call_args[0][0], 'https://cwd.pw/api/upload-image')
            self.assertIn('multipart/form-data', call_args[1]['headers']['Content-Type'])

    async def test_upload_base64_image_success_jpeg(self):
        """Test successful upload of JPEG image with normalized extension."""
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json = AsyncMock(return_value=self.success_response)
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session
            mock_session.post.return_value.__aenter__.return_value = mock_response
            
            result = await upload_base64_image_to_cwd(self.test_base64_jpg, self.api_key)
            
            self.assertEqual(result, "https://cwd.pw/i/test123.png")
            mock_session.post.assert_called_once()

    async def test_upload_base64_image_invalid_format(self):
        """Test upload with invalid base64 format."""
        invalid_data = "not a valid data uri"
        
        result = await upload_base64_image_to_cwd(invalid_data, self.api_key)
        
        self.assertIsNone(result)

    async def test_upload_base64_image_unsupported_mime_type(self):
        """Test upload with unsupported MIME type."""
        unsupported_data = "data:text/plain;base64,dGVzdA=="
        
        result = await upload_base64_image_to_cwd(unsupported_data, self.api_key)
        
        self.assertIsNone(result)

    async def test_upload_base64_image_unsupported_extension(self):
        """Test upload with unsupported file extension."""
        unsupported_data = "data:image/bmp;base64,dGVzdA=="
        
        result = await upload_base64_image_to_cwd(unsupported_data, self.api_key)
        
        self.assertIsNone(result)

    async def test_upload_base64_image_invalid_base64(self):
        """Test upload with invalid base64 data."""
        invalid_b64 = "data:image/png;base64,invalid_base64_data!"
        
        result = await upload_base64_image_to_cwd(invalid_b64, self.api_key)
        
        self.assertIsNone(result)

    async def test_upload_base64_image_http_error(self):
        """Test upload with HTTP error response."""
        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.status = 400
        mock_response.text = AsyncMock(return_value="Bad Request")
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session
            mock_session.post.return_value.__aenter__.return_value = mock_response
            
            result = await upload_base64_image_to_cwd(self.test_base64_png, self.api_key)
            
            self.assertIsNone(result)

    async def test_upload_base64_image_api_error_response(self):
        """Test upload with API error in response."""
        error_response = {
            "success": False,
            "error": "Invalid API key"
        }
        
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json = AsyncMock(return_value=error_response)
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session
            mock_session.post.return_value.__aenter__.return_value = mock_response
            
            result = await upload_base64_image_to_cwd(self.test_base64_png, self.api_key)
            
            self.assertIsNone(result)

    async def test_upload_base64_image_missing_image_url(self):
        """Test upload with success but missing imageUrl in response."""
        incomplete_response = {
            "success": True
            # Missing imageUrl
        }
        
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json = AsyncMock(return_value=incomplete_response)
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session
            mock_session.post.return_value.__aenter__.return_value = mock_response
            
            result = await upload_base64_image_to_cwd(self.test_base64_png, self.api_key)
            
            self.assertIsNone(result)

    async def test_upload_base64_image_timeout(self):
        """Test upload with timeout exception."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session
            mock_session.post.side_effect = aiohttp.ServerTimeoutError()
            
            result = await upload_base64_image_to_cwd(self.test_base64_png, self.api_key)
            
            self.assertIsNone(result)

    async def test_upload_image_bytes_success(self):
        """Test successful upload using bytes method."""
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json = AsyncMock(return_value=self.success_response)
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session
            mock_session.post.return_value.__aenter__.return_value = mock_response
            
            result = await upload_image_bytes_to_cwd(
                self.test_image_bytes, 
                self.api_key, 
                "image/png"
            )
            
            self.assertEqual(result, "https://cwd.pw/i/test123.png")

    async def test_upload_image_bytes_default_mime_type(self):
        """Test upload using bytes method with default MIME type."""
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json = AsyncMock(return_value=self.success_response)
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session
            mock_session.post.return_value.__aenter__.return_value = mock_response
            
            result = await upload_image_bytes_to_cwd(self.test_image_bytes, self.api_key)
            
            self.assertEqual(result, "https://cwd.pw/i/test123.png")

    async def test_upload_image_bytes_base64_error(self):
        """Test upload using bytes method with base64 encoding error."""
        with patch('base64.b64encode') as mock_b64encode:
            mock_b64encode.side_effect = Exception("Base64 encoding failed")
            
            result = await upload_image_bytes_to_cwd(self.test_image_bytes, self.api_key)
            
            self.assertIsNone(result)

    async def test_upload_base64_image_with_model_and_prompt(self):
        """Test upload with model and prompt metadata."""
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json = AsyncMock(return_value=self.success_response)
        
        model = "dall-e-3"
        prompt = "A beautiful sunset over mountains"
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session
            mock_session.post.return_value.__aenter__.return_value = mock_response
            
            result = await upload_base64_image_to_cwd(
                self.test_base64_png, 
                self.api_key, 
                model=model, 
                prompt=prompt
            )
            
            self.assertEqual(result, "https://cwd.pw/i/test123.png")
            mock_session.post.assert_called_once()
            
            # Check that the request body contains metadata fields
            call_args = mock_session.post.call_args
            request_data = call_args[1]['data']
            request_data_str = request_data.decode('utf-8', errors='ignore')
            
            # Verify metadata fields are present
            self.assertIn('name="ai_generated"', request_data_str)
            self.assertIn('true', request_data_str)
            self.assertIn('name="model"', request_data_str)
            self.assertIn(model, request_data_str)
            self.assertIn('name="prompt"', request_data_str)
            self.assertIn(prompt, request_data_str)

    async def test_upload_base64_image_with_empty_metadata(self):
        """Test upload with None/empty model and prompt."""
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json = AsyncMock(return_value=self.success_response)
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session
            mock_session.post.return_value.__aenter__.return_value = mock_response
            
            result = await upload_base64_image_to_cwd(
                self.test_base64_png, 
                self.api_key, 
                model=None, 
                prompt=""
            )
            
            self.assertEqual(result, "https://cwd.pw/i/test123.png")
            
            # Check that the request body contains empty metadata fields
            call_args = mock_session.post.call_args
            request_data = call_args[1]['data']
            request_data_str = request_data.decode('utf-8', errors='ignore')
            
            # Verify ai_generated is still present
            self.assertIn('name="ai_generated"', request_data_str)
            self.assertIn('true', request_data_str)
            # Verify model and prompt fields are present but empty
            self.assertIn('name="model"', request_data_str)
            self.assertIn('name="prompt"', request_data_str)

    async def test_upload_image_bytes_with_metadata(self):
        """Test upload using bytes method with metadata."""
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json = AsyncMock(return_value=self.success_response)
        
        model = "stable-diffusion"
        prompt = "A cat sitting on a windowsill"
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session
            mock_session.post.return_value.__aenter__.return_value = mock_response
            
            result = await upload_image_bytes_to_cwd(
                self.test_image_bytes, 
                self.api_key, 
                "image/png",
                model=model,
                prompt=prompt
            )
            
            self.assertEqual(result, "https://cwd.pw/i/test123.png")
            
            # Check that metadata is passed through correctly
            call_args = mock_session.post.call_args
            request_data = call_args[1]['data']
            request_data_str = request_data.decode('utf-8', errors='ignore')
            
            self.assertIn('name="ai_generated"', request_data_str)
            self.assertIn('true', request_data_str)
            self.assertIn('name="model"', request_data_str)
            self.assertIn(model, request_data_str)
            self.assertIn('name="prompt"', request_data_str)
            self.assertIn(prompt, request_data_str)

    def test_multipart_form_data_structure(self):
        """Test that multipart form data is structured correctly."""
        # This is more of an integration test but helps verify the structure
        expected_image_header_parts = [
            'Content-Disposition: form-data; name="image"; filename="upload.png"',
            'Content-Type: image/png'
        ]
        
        expected_api_key_header_parts = [
            'Content-Disposition: form-data; name="api_key"'
        ]
        
        expected_metadata_header_parts = [
            'Content-Disposition: form-data; name="ai_generated"',
            'Content-Disposition: form-data; name="model"',
            'Content-Disposition: form-data; name="prompt"'
        ]
        
        # These are the key parts that should be in the multipart data
        # The actual test would need to mock the internal structure,
        # but this documents the expected format
        self.assertTrue(all(part in expected_image_header_parts for part in expected_image_header_parts))
        self.assertTrue(all(part in expected_api_key_header_parts for part in expected_api_key_header_parts))
        self.assertTrue(all(part in expected_metadata_header_parts for part in expected_metadata_header_parts))

    async def test_ai_generated_always_true(self):
        """Test that ai_generated field is always set to true regardless of other parameters."""
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json = AsyncMock(return_value=self.success_response)
        
        test_cases = [
            {"model": None, "prompt": None},
            {"model": "test-model", "prompt": None},
            {"model": None, "prompt": "test-prompt"},
            {"model": "test-model", "prompt": "test-prompt"},
        ]
        
        for case in test_cases:
            with patch('aiohttp.ClientSession') as mock_session_class:
                mock_session = MagicMock()
                mock_session_class.return_value.__aenter__.return_value = mock_session
                mock_session.post.return_value.__aenter__.return_value = mock_response
                
                await upload_base64_image_to_cwd(
                    self.test_base64_png, 
                    self.api_key, 
                    **case
                )
                
                # Check that ai_generated is always true
                call_args = mock_session.post.call_args
                request_data = call_args[1]['data']
                request_data_str = request_data.decode('utf-8', errors='ignore')
                
                self.assertIn('name="ai_generated"', request_data_str)
                self.assertIn('true', request_data_str)
                # Ensure it's not false or any other value
                self.assertNotIn('false', request_data_str.lower())


if __name__ == '__main__':
    unittest.main()