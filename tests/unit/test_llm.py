import unittest
from typing import Optional  # Added List for type hinting
from unittest.mock import MagicMock, patch

from google.genai import client as client_library  # Changed to google.genai
from google.genai import types as genai_types  # Changed to google.genai
from google.longrunning import operations_pb2  # For specing Operation
from google.rpc import status_pb2  # For specing Operation.error

# Assuming llm.py is in bot.llm
from bot.llm import (
    GEMINI_API_KEY,
    GEMINI_VIDEO_MODEL,
    generate_image_with_gemini,
    generate_image_with_vertex,  # Added for the new tests
    generate_video_with_veo,
)

# Constants like VERTEX_IMAGE_MODEL, VERTEX_PROJECT_ID, VERTEX_LOCATION are imported by llm.py itself
# and will be patched in the tests for generate_image_with_vertex.

# Helper to create a mock operation
def _create_mock_operation(name: str, done: bool, error_code: Optional[int] = None, response_videos: Optional[list] = None):
    op = MagicMock(spec=client_library.Operation)
    op.operation = MagicMock(spec=operations_pb2.Operation) # Mocking the nested 'operation' attribute
    op.operation.name = name
    op.done.return_value = done # Set the return value for the method call

    if error_code is not None:
        op.error = MagicMock(spec=status_pb2.Status)
        op.error.code = error_code
        op.error.message = f"Mock error code {error_code}"
    else:
        op.error = None # Explicitly set error to None for success cases

    if response_videos is not None:
        op.response = MagicMock(spec=genai_types.GenerateVideosResponse)
        op.response.generated_videos = response_videos
    else:
        op.response = None
    return op


class TestGenerateVideoWithVeoNew(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        patcher = patch.dict('os.environ', {'GEMINI_API_KEY': 'test_key'})
        self.mock_env = patcher.start()
        self.addCleanup(patcher.stop)

    @patch('bot.llm.time.sleep', return_value=None) 
    @patch('bot.llm.detect_mime_type', return_value="image/jpeg")
    @patch('bot.llm.genai.Client') 
    async def test_generate_video_success_with_image(self, MockGeminiClient, mock_detect_mime, mock_sleep):
        mock_sync_client = MockGeminiClient.return_value 

        op_name = "operations/test-op-123"
        initial_op = _create_mock_operation(name=op_name, done=False)
        mock_sync_client.models.generate_videos.return_value = initial_op

        polling_op_not_done = _create_mock_operation(name=op_name, done=False)
        
        mock_video_resource = MagicMock(spec=genai_types.VideoResource) 
        mock_video_resource.name = "files/generated-video-file-id"
        mock_video_resource.mime_type = "video/mp4" # Added mime_type here
        
        mock_generated_video_entry = MagicMock(spec=genai_types.GeneratedVideo) 
        mock_generated_video_entry.video = mock_video_resource 

        final_op_success = _create_mock_operation(name=op_name, done=True, response_videos=[mock_generated_video_entry])
        
        mock_sync_client.operations.get.side_effect = [polling_op_not_done, final_op_success]

        # No need to mock files.get anymore if mime_type is on VideoResource
        # mock_file_meta = MagicMock(spec=genai_types.File)
        # mock_file_meta.mime_type = "video/mp4"
        # mock_file_meta.name = "files/generated-video-file-id"
        # mock_sync_client.files.get.return_value = mock_file_meta

        mock_download_response_content = b"mock_video_content" # Changed to content directly
        mock_sync_client.files.download.return_value = mock_download_response_content


        system_prompt = "System prompt"
        user_prompt = "Make a video of a cat"
        image_data = b"fake_image_bytes"

        video_bytes, mime_type = await generate_video_with_veo(system_prompt, user_prompt, image_data)

        self.assertEqual(video_bytes, b"mock_video_content")
        self.assertEqual(mime_type, "video/mp4")

        MockGeminiClient.assert_called_once_with(api_key=GEMINI_API_KEY)
        mock_detect_mime.assert_called_once_with(image_data)
        
        mock_sync_client.models.generate_videos.assert_called_once()
        call_args, call_kwargs = mock_sync_client.models.generate_videos.call_args
        self.assertEqual(call_kwargs['model'], GEMINI_VIDEO_MODEL)
        # The prompt in generate_video_with_veo is just user_prompt now
        self.assertEqual(call_kwargs['prompt'], user_prompt) 
        self.assertIsNotNone(call_kwargs['image'])
        # The image part is constructed directly in generate_video_with_veo, not a types.Part
        self.assertEqual(call_kwargs['image'], image_data) 
        self.assertIsInstance(call_kwargs['config'], genai_types.GenerateVideosConfig)
        # Safety settings are not directly passed in generate_videos in llm.py
        # self.assertEqual(call_kwargs['safety_settings'], _safety_settings)

        self.assertEqual(mock_sync_client.operations.get.call_count, 2)
        mock_sync_client.operations.get.assert_any_call(initial_op) # Pass the operation object

        mock_sleep.assert_called_once_with(20)

        # mock_sync_client.files.get.assert_called_once_with(name="files/generated-video-file-id")
        mock_sync_client.files.download.assert_called_once_with(file=mock_video_resource) # Pass the resource


    @patch('bot.llm.time.sleep', return_value=None)
    @patch('bot.llm.detect_mime_type') 
    @patch('bot.llm.genai.Client')
    async def test_generate_video_success_prompt_only(self, MockGeminiClient, mock_detect_mime, mock_sleep):
        mock_sync_client = MockGeminiClient.return_value
        op_name = "operations/prompt-op-456"
        initial_op = _create_mock_operation(name=op_name, done=False)
        mock_sync_client.models.generate_videos.return_value = initial_op

        mock_video_resource = MagicMock(spec=genai_types.VideoResource, name="files/prompt-video-id", mime_type="video/webm")
        mock_generated_video_entry = MagicMock(spec=genai_types.GeneratedVideo, video=mock_video_resource)
        final_op_success = _create_mock_operation(name=op_name, done=True, response_videos=[mock_generated_video_entry])
        mock_sync_client.operations.get.return_value = final_op_success 

        mock_download_response_content = b"prompt_video_data"
        mock_sync_client.files.download.return_value = mock_download_response_content


        system_prompt = "Sys"
        user_prompt = "Video please"
        video_bytes, mime_type = await generate_video_with_veo(system_prompt, user_prompt, image_data=None)

        self.assertEqual(video_bytes, b"prompt_video_data")
        self.assertEqual(mime_type, "video/webm")
        mock_detect_mime.assert_not_called()
        call_args, call_kwargs = mock_sync_client.models.generate_videos.call_args
        self.assertIsNone(call_kwargs['image']) 
        mock_sleep.assert_called_once_with(20)
        mock_sync_client.operations.get.assert_called_once_with(initial_op)
        mock_sync_client.files.download.assert_called_once_with(file=mock_video_resource)


    @patch('bot.llm.time.sleep', return_value=None)
    @patch('bot.llm.genai.Client')
    async def test_generate_video_operation_ok_no_videos(self, MockGeminiClient, mock_sleep):
        mock_sync_client = MockGeminiClient.return_value
        op_name = "operations/no-video-op-789"
        initial_op = _create_mock_operation(name=op_name, done=False)
        mock_sync_client.models.generate_videos.return_value = initial_op
        
        final_op_no_video = _create_mock_operation(name=op_name, done=True, response_videos=[])
        mock_sync_client.operations.get.return_value = final_op_no_video

        video_bytes, mime_type = await generate_video_with_veo("S", "U", image_data=None)

        self.assertIsNone(video_bytes)
        self.assertIsNone(mime_type)
        mock_sync_client.files.download.assert_not_called()

    @patch('bot.llm.genai.Client')
    async def test_generate_video_generate_videos_raises_exception(self, MockGeminiClient):
        mock_sync_client = MockGeminiClient.return_value
        mock_sync_client.models.generate_videos.side_effect = Exception("Generate videos API call failed")

        with patch('bot.llm.logger.error') as mock_logger_error:
            video_bytes, mime_type = await generate_video_with_veo("S", "U", image_data=None)

            self.assertIsNone(video_bytes)
            self.assertIsNone(mime_type)
            mock_logger_error.assert_any_call(
                "Exception during VEO video generation or polling: Generate videos API call failed", exc_info=True
            )

    @patch('bot.llm.time.sleep', return_value=None)
    @patch('bot.llm.genai.Client')
    async def test_generate_video_operation_fails_with_error(self, MockGeminiClient, mock_sleep):
        mock_sync_client = MockGeminiClient.return_value
        op_name = "operations/op-error-000"
        initial_op = _create_mock_operation(name=op_name, done=False)
        mock_sync_client.models.generate_videos.return_value = initial_op

        final_op_with_error = _create_mock_operation(name=op_name, done=True, error_code=3)
        mock_sync_client.operations.get.return_value = final_op_with_error
        
        with patch('bot.llm.logger.error') as mock_logger_error:
            video_bytes, mime_type = await generate_video_with_veo("S", "U", image_data=None)

            self.assertIsNone(video_bytes)
            self.assertIsNone(mime_type)
            mock_logger_error.assert_any_call(
                f"Video generation operation failed with error: {final_op_with_error.error}"
            )

    @patch('bot.llm.time.sleep', return_value=None)
    @patch('bot.llm.genai.Client')
    async def test_generate_video_polling_timeout(self, MockGeminiClient, mock_sleep):
        mock_sync_client = MockGeminiClient.return_value
        op_name = "operations/timeout-op-111"
        
        persistent_not_done_op = _create_mock_operation(name=op_name, done=False)
        mock_sync_client.models.generate_videos.return_value = persistent_not_done_op
        mock_sync_client.operations.get.return_value = persistent_not_done_op

        expected_sleep_calls = 30 
        
        with patch('bot.llm.logger.error') as mock_logger_error:
            video_bytes, mime_type = await generate_video_with_veo("S", "U", image_data=None)

            self.assertIsNone(video_bytes)
            self.assertIsNone(mime_type)
            self.assertEqual(mock_sleep.call_count, expected_sleep_calls)
            mock_logger_error.assert_any_call(
                f"Video generation operation timed out after {expected_sleep_calls * 20} seconds."
            )

    @patch('bot.llm.time.sleep', return_value=None)
    @patch('bot.llm.genai.Client')
    async def test_generate_video_operations_get_raises_exception(self, MockGeminiClient, mock_sleep):
        mock_sync_client = MockGeminiClient.return_value
        op_name = "operations/op-get-fail-222"
        initial_op = _create_mock_operation(name=op_name, done=False)
        mock_sync_client.models.generate_videos.return_value = initial_op

        mock_sync_client.operations.get.side_effect = Exception("operations.get API call failed")

        with patch('bot.llm.logger.error') as mock_logger_error:
            video_bytes, mime_type = await generate_video_with_veo("S", "U", image_data=None)
            self.assertIsNone(video_bytes)
            self.assertIsNone(mime_type)
            mock_logger_error.assert_any_call(
                 "Exception during VEO video generation or polling: operations.get API call failed", exc_info=True
            )

    @patch('bot.llm.time.sleep', return_value=None)
    @patch('bot.llm.genai.Client')
    async def test_generate_video_files_download_raises_exception(self, MockGeminiClient, mock_sleep):
        mock_sync_client = MockGeminiClient.return_value
        op_name = "operations/files-download-fail-444"
        initial_op = _create_mock_operation(name=op_name, done=False)
        mock_sync_client.models.generate_videos.return_value = initial_op

        mock_video_resource = MagicMock(spec=genai_types.VideoResource, name="files/video-id-for-files-download-fail", mime_type="video/mp4")
        mock_generated_video_entry = MagicMock(spec=genai_types.GeneratedVideo, video=mock_video_resource)
        final_op_success = _create_mock_operation(name=op_name, done=True, response_videos=[mock_generated_video_entry])
        mock_sync_client.operations.get.return_value = final_op_success
        
        mock_sync_client.files.download.side_effect = Exception("files.download API call failed")

        with patch('bot.llm.logger.error') as mock_logger_error:
            video_bytes, mime_type = await generate_video_with_veo("S", "U", image_data=None)
            self.assertIsNone(video_bytes)
            self.assertIsNone(mime_type)
            mock_logger_error.assert_any_call(
                "Exception during VEO video generation or polling: files.download API call failed", exc_info=True
            )

    @patch('bot.llm.time.sleep', return_value=None)
    @patch('bot.llm.detect_mime_type', return_value="application/pdf") 
    @patch('bot.llm.genai.Client')
    async def test_generate_video_invalid_image_mime_logs_error_and_skips_image(self, MockGeminiClient, mock_detect_mime, mock_sleep):
        mock_sync_client = MockGeminiClient.return_value
        op_name = "operations/invalid-mime-op-555"
        initial_op = _create_mock_operation(name=op_name, done=False)
        mock_sync_client.models.generate_videos.return_value = initial_op

        mock_video_resource = MagicMock(spec=genai_types.VideoResource, name="files/video-no-image-id", mime_type="video/mp4")
        mock_generated_video_entry = MagicMock(spec=genai_types.GeneratedVideo, video=mock_video_resource)
        final_op_success = _create_mock_operation(name=op_name, done=True, response_videos=[mock_generated_video_entry])
        mock_sync_client.operations.get.return_value = final_op_success
        
        mock_download_response_content = b"video_content_no_image"
        mock_sync_client.files.download.return_value = mock_download_response_content

        image_data = b"fake_pdf_bytes"
        with patch('bot.llm.logger.error') as mock_logger_error:
            video_bytes, mime_type = await generate_video_with_veo("S", "U", image_data=image_data)

        self.assertEqual(video_bytes, b"video_content_no_image")
        self.assertEqual(mime_type, "video/mp4")
        
        mock_detect_mime.assert_called_once_with(image_data)
        mock_logger_error.assert_any_call("Invalid MIME type for image_data: application/pdf. Skipping image.")
        
        call_args, call_kwargs = mock_sync_client.models.generate_videos.call_args
        self.assertIsNone(call_kwargs['image'])


# Tests for generate_image_with_vertex (refactored to use get_vertex_client)
@patch('bot.llm.get_vertex_client') # Main mock target
@patch('bot.llm.Image')             # PIL.Image, still used
@patch('bot.llm.BytesIO')           # io.BytesIO, still used
class TestGenerateImageWithVertex(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.mock_vertex_image_model_name = patch('bot.llm.VERTEX_IMAGE_MODEL', 'test-imagen-model@005')
        self.active_image_model_patch = self.mock_vertex_image_model_name.start()
        self.addCleanup(self.mock_vertex_image_model_name.stop)

    async def test_successful_generation_multiple_images_default(self, MockBytesIO, MockPILImage, MockGetVertexClient):
        """Test successful generation of 4 images (default) using genai.Client."""
        mock_vertex_client_instance = MockGetVertexClient.return_value
        self.assertIsNotNone(mock_vertex_client_instance, "get_vertex_client mock should return a client instance")
        mock_generative_model_instance = mock_vertex_client_instance.generative_model.return_value
        self.assertIsNotNone(mock_generative_model_instance, "client.generative_model mock should return a model instance")
        
        mock_candidates = []
        for i in range(4): 
            mock_part = MagicMock()
            mock_part.inline_data = MagicMock()
            mock_part.inline_data.data = f"raw_image_bytes_{i}".encode('utf-8')
            mock_part.inline_data.mime_type = "image/png" 
            
            mock_content = MagicMock()
            mock_content.parts = [mock_part]
            
            mock_candidate = MagicMock()
            mock_candidate.content = mock_content
            mock_candidates.append(mock_candidate)
            
        mock_response = MagicMock()
        mock_response.candidates = mock_candidates
        mock_generative_model_instance.generate_content.return_value = mock_response

        mock_pil_img_instance = MockPILImage.open.return_value
        mock_pil_img_instance.mode = 'RGB'
        
        mock_bytes_io_instance = MockBytesIO.return_value
        mock_bytes_io_instance.getvalue.return_value = b"processed_jpeg_bytes"

        prompt = "a beautiful landscape"
        result_images = await generate_image_with_vertex(prompt=prompt) 

        self.assertEqual(len(result_images), 4)
        for i in range(4):
            self.assertEqual(result_images[i], b"processed_jpeg_bytes")

        MockGetVertexClient.assert_called_once()
        mock_vertex_client_instance.generative_model.assert_called_once_with(model_name='test-imagen-model@005')
        
        # Check that generation_config was an instance of types.GenerationConfig
        args, kwargs = mock_generative_model_instance.generate_content.call_args
        self.assertEqual(kwargs['contents'], prompt)
        self.assertIsInstance(kwargs['generation_config'], genai_types.GenerationConfig)
        self.assertEqual(kwargs['generation_config'].candidate_count, 4)
        
        self.assertEqual(MockPILImage.open.call_count, 4)
        self.assertEqual(mock_pil_img_instance.save.call_count, 4)


    async def test_successful_generation_specific_number_of_images(self, MockBytesIO, MockPILImage, MockGetVertexClient):
        """Test successful generation of 2 images using genai.Client."""
        num_images = 2
        mock_vertex_client_instance = MockGetVertexClient.return_value
        mock_generative_model_instance = mock_vertex_client_instance.generative_model.return_value
        
        mock_candidates = []
        for i in range(num_images):
            mock_part = MagicMock(inline_data=MagicMock(data=f"img_bytes_{i}".encode(), mime_type="image/png"))
            mock_candidates.append(MagicMock(content=MagicMock(parts=[mock_part])))
            
        mock_response = MagicMock(candidates=mock_candidates)
        mock_generative_model_instance.generate_content.return_value = mock_response

        mock_pil_img_instance = MockPILImage.open.return_value
        mock_pil_img_instance.mode = 'RGBA' 
        
        mock_bytes_io_instance = MockBytesIO.return_value
        mock_bytes_io_instance.getvalue.return_value = b"converted_jpeg"

        prompt = "a cat playing guitar"
        result_images = await generate_image_with_vertex(prompt=prompt, number_of_images=num_images)

        self.assertEqual(len(result_images), num_images)
        self.assertEqual(result_images[0], b"converted_jpeg")

        mock_vertex_client_instance.generative_model.assert_called_once_with(model_name='test-imagen-model@005')
        args, kwargs = mock_generative_model_instance.generate_content.call_args
        self.assertEqual(kwargs['contents'], prompt)
        self.assertIsInstance(kwargs['generation_config'], genai_types.GenerationConfig)
        self.assertEqual(kwargs['generation_config'].candidate_count, num_images)

        self.assertEqual(MockPILImage.open.call_count, num_images)
        self.assertEqual(mock_pil_img_instance.convert.call_count, num_images)
        mock_pil_img_instance.convert.assert_called_with('RGB')


    async def test_api_error_generate_content_raises_exception(self, MockBytesIO, MockPILImage, MockGetVertexClient):
        """Test handling of API error when model.generate_content raises an exception."""
        mock_vertex_client_instance = MockGetVertexClient.return_value
        mock_generative_model_instance = mock_vertex_client_instance.generative_model.return_value
        mock_generative_model_instance.generate_content.side_effect = Exception("Vertex GenAI API Error")

        prompt = "an error prompt"
        result_images = await generate_image_with_vertex(prompt=prompt)

        self.assertEqual(len(result_images), 0)
        MockGetVertexClient.assert_called_once()
        mock_vertex_client_instance.generative_model.assert_called_once_with(model_name='test-imagen-model@005')
        mock_generative_model_instance.generate_content.assert_called_once() 
        MockPILImage.open.assert_not_called()


    async def test_no_candidates_returned_in_response(self, MockBytesIO, MockPILImage, MockGetVertexClient):
        """Test handling of an empty candidates list in the API response."""
        mock_vertex_client_instance = MockGetVertexClient.return_value
        mock_generative_model_instance = mock_vertex_client_instance.generative_model.return_value
        
        mock_response = MagicMock()
        mock_response.candidates = [] 
        mock_generative_model_instance.generate_content.return_value = mock_response

        prompt = "a prompt that yields no candidates"
        result_images = await generate_image_with_vertex(prompt=prompt)

        self.assertEqual(len(result_images), 0)
        mock_generative_model_instance.generate_content.assert_called_once()
        MockPILImage.open.assert_not_called()

    async def test_pil_conversion_failure_one_image(self, MockBytesIO, MockPILImage, MockGetVertexClient):
        """Test PIL conversion failure for one of the images."""
        mock_vertex_client_instance = MockGetVertexClient.return_value
        mock_generative_model_instance = mock_vertex_client_instance.generative_model.return_value

        mock_candidates = []
        for i in range(2):
            mock_part = MagicMock(inline_data=MagicMock(data=f"pil_fail_bytes_{i}".encode(), mime_type="image/png"))
            mock_candidates.append(MagicMock(content=MagicMock(parts=[mock_part])))
        
        mock_response = MagicMock(candidates=mock_candidates)
        mock_generative_model_instance.generate_content.return_value = mock_response

        mock_pil_img_instance_good = MagicMock()
        mock_pil_img_instance_good.mode = 'RGB'
        
        mock_pil_img_instance_bad = MagicMock()
        mock_pil_img_instance_bad.mode = 'RGB'
        mock_pil_img_instance_bad.save.side_effect = Exception("PIL save failed")

        MockPILImage.open.side_effect = [mock_pil_img_instance_good, mock_pil_img_instance_bad]
        
        mock_good_bytes_io_instance = MagicMock()
        mock_good_bytes_io_instance.getvalue.return_value = b"good_pil_image_bytes"
        MockBytesIO.side_effect = [mock_good_bytes_io_instance, MagicMock()]

        prompt = "test PIL failure"
        result_images = await generate_image_with_vertex(prompt=prompt, number_of_images=2)

        self.assertEqual(len(result_images), 1) 
        if result_images:
            self.assertEqual(result_images[0], b"good_pil_image_bytes") 

        self.assertEqual(MockPILImage.open.call_count, 2)
        self.assertEqual(mock_pil_img_instance_good.save.call_count, 1)
        self.assertEqual(mock_pil_img_instance_bad.save.call_count, 1)


    async def test_get_vertex_client_returns_none(self, MockBytesIO, MockPILImage, MockGetVertexClient):
        """Test behavior when get_vertex_client returns None."""
        MockGetVertexClient.return_value = None
        
        result_images = await generate_image_with_vertex(prompt="test")
        self.assertEqual(result_images, [])
        MockGetVertexClient.assert_called_once()
        MockPILImage.open.assert_not_called()

    async def test_vertex_image_model_not_configured(self, MockBytesIO, MockPILImage, MockGetVertexClient):
        """Test behavior when VERTEX_IMAGE_MODEL is not configured (None or empty)."""
        self.active_image_model_patch.stop()
        temporary_model_patch = patch('bot.llm.VERTEX_IMAGE_MODEL', None)
        temporary_model_patch.start()
        
        mock_vertex_client_instance = MockGetVertexClient.return_value 

        result_images = await generate_image_with_vertex(prompt="test")
        self.assertEqual(result_images, [])
        
        MockGetVertexClient.assert_called_once() 
        mock_vertex_client_instance.generative_model.assert_not_called() 
        
        temporary_model_patch.stop()
        self.active_image_model_patch.start() 


# Tests for generate_image_with_gemini and CWD upload integration
@patch('bot.llm.get_gemini_client')
@patch('bot.llm.Image')
@patch('bot.llm.BytesIO')
class TestGenerateImageWithGemini(unittest.IsolatedAsyncioTestCase):
    """Test cases for Gemini image generation and CWD upload integration."""

    def setUp(self):
        self.mock_gemini_image_model = patch('bot.llm.GEMINI_IMAGE_MODEL', 'test-gemini-image-model')
        self.active_model_patch = self.mock_gemini_image_model.start()
        self.addCleanup(self.mock_gemini_image_model.stop)

    async def test_generate_image_success_without_upload(self, MockBytesIO, MockPILImage, MockGetGeminiClient):
        """Test successful image generation without CWD upload."""
        mock_client = MockGetGeminiClient.return_value
        
        # Mock response with image data
        mock_inline_data = MagicMock()
        mock_inline_data.data = b"mock_image_data"
        mock_inline_data.mime_type = "image/png"
        
        mock_part = MagicMock()
        mock_part.inline_data = mock_inline_data
        
        mock_content = MagicMock()
        mock_content.parts = [mock_part]
        
        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        
        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        
        mock_client.aio.models.generate_content.return_value = mock_response
        
        # Mock PIL processing
        mock_pil_img = MockPILImage.open.return_value
        mock_pil_img.mode = 'RGB'
        
        mock_bytes_io = MockBytesIO.return_value
        mock_bytes_io.getvalue.return_value = b"processed_jpeg_bytes"
        
        result = await generate_image_with_gemini(
            system_prompt="Test prompt",
            prompt="A test image",
            upload_to_cwd=False
        )
        
        self.assertEqual(result, b"processed_jpeg_bytes")
        mock_client.aio.models.generate_content.assert_called_once()

    @patch('bot.llm.CWD_PW_API_KEY', 'test_api_key')
    @patch('bot.cwd_uploader.upload_image_bytes_to_cwd')
    async def test_generate_image_success_with_upload(self, MockUploadToCwd, MockBytesIO, MockPILImage, MockGetGeminiClient):
        """Test successful image generation with CWD upload."""
        mock_client = MockGetGeminiClient.return_value
        
        # Mock response with image data
        mock_inline_data = MagicMock()
        mock_inline_data.data = b"mock_image_data"
        mock_inline_data.mime_type = "image/png"
        
        mock_part = MagicMock()
        mock_part.inline_data = mock_inline_data
        
        mock_content = MagicMock()
        mock_content.parts = [mock_part]
        
        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        
        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        
        mock_client.aio.models.generate_content.return_value = mock_response
        
        # Mock PIL processing
        mock_pil_img = MockPILImage.open.return_value
        mock_pil_img.mode = 'RGB'
        
        mock_bytes_io = MockBytesIO.return_value
        mock_bytes_io.getvalue.return_value = b"processed_jpeg_bytes"
        
        # Mock successful upload
        MockUploadToCwd.return_value = "https://cwd.pw/i/test123.jpg"
        
        result = await generate_image_with_gemini(
            system_prompt="Test prompt",
            prompt="A test image",
            upload_to_cwd=True
        )
        
        self.assertEqual(result, b"processed_jpeg_bytes")
        MockUploadToCwd.assert_called_once_with(
            image_bytes=b"processed_jpeg_bytes",
            api_key='test_api_key',
            mime_type="image/jpeg",
            model='test-gemini-image-model',
            prompt="A test image"
        )

    @patch('bot.llm.CWD_PW_API_KEY', '')
    async def test_generate_image_no_upload_when_no_api_key(self, MockBytesIO, MockPILImage, MockGetGeminiClient):
        """Test that upload is skipped when no API key is configured."""
        mock_client = MockGetGeminiClient.return_value
        
        # Mock response with image data
        mock_inline_data = MagicMock()
        mock_inline_data.data = b"mock_image_data"
        mock_inline_data.mime_type = "image/png"
        
        mock_part = MagicMock()
        mock_part.inline_data = mock_inline_data
        
        mock_content = MagicMock()
        mock_content.parts = [mock_part]
        
        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        
        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        
        mock_client.aio.models.generate_content.return_value = mock_response
        
        # Mock PIL processing
        mock_pil_img = MockPILImage.open.return_value
        mock_pil_img.mode = 'RGB'
        
        mock_bytes_io = MockBytesIO.return_value
        mock_bytes_io.getvalue.return_value = b"processed_jpeg_bytes"
        
        with patch('bot.cwd_uploader.upload_image_bytes_to_cwd') as MockUpload:
            result = await generate_image_with_gemini(
                system_prompt="Test prompt",
                prompt="A test image",
                upload_to_cwd=True
            )
            
            self.assertEqual(result, b"processed_jpeg_bytes")
            MockUpload.assert_not_called()

    @patch('bot.llm.CWD_PW_API_KEY', 'test_api_key')
    @patch('bot.cwd_uploader.upload_image_bytes_to_cwd')
    async def test_generate_image_upload_failure_does_not_affect_result(self, MockUploadToCwd, MockBytesIO, MockPILImage, MockGetGeminiClient):
        """Test that upload failure doesn't affect image generation result."""
        mock_client = MockGetGeminiClient.return_value
        
        # Mock response with image data
        mock_inline_data = MagicMock()
        mock_inline_data.data = b"mock_image_data"
        mock_inline_data.mime_type = "image/png"
        
        mock_part = MagicMock()
        mock_part.inline_data = mock_inline_data
        
        mock_content = MagicMock()
        mock_content.parts = [mock_part]
        
        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        
        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        
        mock_client.aio.models.generate_content.return_value = mock_response
        
        # Mock PIL processing
        mock_pil_img = MockPILImage.open.return_value
        mock_pil_img.mode = 'RGB'
        
        mock_bytes_io = MockBytesIO.return_value
        mock_bytes_io.getvalue.return_value = b"processed_jpeg_bytes"
        
        # Mock upload failure
        MockUploadToCwd.side_effect = Exception("Upload failed")
        
        result = await generate_image_with_gemini(
            system_prompt="Test prompt",
            prompt="A test image",
            upload_to_cwd=True
        )
        
        # Image generation should still succeed despite upload failure
        self.assertEqual(result, b"processed_jpeg_bytes")
        MockUploadToCwd.assert_called_once()

    async def test_generate_image_no_image_data_in_response(self, MockBytesIO, MockPILImage, MockGetGeminiClient):
        """Test handling when no image data is found in response."""
        mock_client = MockGetGeminiClient.return_value
        
        # Mock response without image data
        mock_part = MagicMock()
        mock_part.text = "No image generated"
        # No inline_data attribute
        
        mock_content = MagicMock()
        mock_content.parts = [mock_part]
        
        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        
        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        
        mock_client.aio.models.generate_content.return_value = mock_response
        
        result = await generate_image_with_gemini(
            system_prompt="Test prompt",
            prompt="A test image"
        )
        
        self.assertIsNone(result)
        MockPILImage.open.assert_not_called()


# Tests for Vertex image generation with CWD upload
@patch('bot.llm.get_vertex_client')
@patch('bot.llm.Image')
@patch('bot.llm.BytesIO')
class TestVertexImageWithCwdUpload(unittest.IsolatedAsyncioTestCase):
    """Test CWD upload integration with Vertex image generation."""

    def setUp(self):
        self.mock_vertex_image_model = patch('bot.llm.VERTEX_IMAGE_MODEL', 'test-imagen-model@005')
        self.active_model_patch = self.mock_vertex_image_model.start()
        self.addCleanup(self.mock_vertex_image_model.stop)

    @patch('bot.llm.CWD_PW_API_KEY', 'test_api_key')
    @patch('bot.cwd_uploader.upload_image_bytes_to_cwd')
    async def test_vertex_image_generation_with_upload(self, MockUploadToCwd, MockBytesIO, MockPILImage, MockGetVertexClient):
        """Test Vertex image generation with successful CWD upload."""
        mock_vertex_client = MockGetVertexClient.return_value
        
        # Mock successful image generation
        mock_image = MagicMock()
        mock_image.image_bytes = b"vertex_image_data"
        mock_image.mime_type = "image/png"
        
        mock_generated_image = MagicMock()
        mock_generated_image.image = mock_image
        mock_generated_image.enhanced_prompt = None
        
        mock_response = MagicMock()
        mock_response.generated_images = [mock_generated_image]
        
        mock_vertex_client.aio.models.generate_images.return_value = mock_response
        
        # Mock PIL processing
        mock_pil_img = MockPILImage.open.return_value
        mock_pil_img.mode = 'RGB'
        
        mock_bytes_io = MockBytesIO.return_value
        mock_bytes_io.getvalue.return_value = b"processed_jpeg_bytes"
        
        # Mock successful upload
        MockUploadToCwd.return_value = "https://cwd.pw/i/vertex123.jpg"
        
        result = await generate_image_with_vertex(
            prompt="A test image",
            number_of_images=1,
            upload_to_cwd=True
        )
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], b"processed_jpeg_bytes")
        MockUploadToCwd.assert_called_once_with(
            image_bytes=b"processed_jpeg_bytes",
            api_key='test_api_key',
            mime_type="image/jpeg",
            model='test-imagen-model@005',
            prompt="A test image"
        )


# This allows running the test file directly.
if __name__ == '__main__':
    unittest.main()
