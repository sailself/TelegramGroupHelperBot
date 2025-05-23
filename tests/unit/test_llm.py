import asyncio
import unittest
from unittest.mock import MagicMock, patch, call
import time # For type hinting if needed, but time.sleep is mocked

from google.longrunning import operations_pb2 # For specing Operation
from google.rpc import status_pb2 # For specing Operation.error
from google.genai import client as client_library # Changed to google.genai
from google.genai import types as genai_types # Changed to google.genai

# Assuming llm.py is in bot.llm
from bot.llm import generate_video_with_veo, GEMINI_VIDEO_MODEL, _safety_settings, GEMINI_API_KEY
# GEMINI_MAX_OUTPUT_TOKENS, GEMINI_TEMPERATURE, GEMINI_TOP_P, GEMINI_TOP_K are not directly used by generate_videos config

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
        # Ensure GEMINI_API_KEY is available for genai.Client instantiation within the function
        # This is only needed if the client isn't fully mocked out before instantiation.
        # Given we patch 'bot.llm.genai.Client', this might not be strictly necessary
        # but good for clarity.
        patcher = patch.dict('os.environ', {'GEMINI_API_KEY': 'test_key'})
        self.mock_env = patcher.start()
        self.addCleanup(patcher.stop)

    @patch('bot.llm.time.sleep', return_value=None) # Prevent actual sleep
    @patch('bot.llm.detect_mime_type', return_value="image/jpeg")
    @patch('bot.llm.genai.Client') # Mock the client constructor
    async def test_generate_video_success_with_image(self, MockGeminiClient, mock_detect_mime, mock_sleep):
        mock_sync_client = MockGeminiClient.return_value # This is the instance used in _sync_generate_video

        # 1. Initial operation from generate_videos
        op_name = "operations/test-op-123"
        initial_op = _create_mock_operation(name=op_name, done=False)
        mock_sync_client.models.generate_videos.return_value = initial_op

        # 2. Polled operation - still not done
        polling_op_not_done = _create_mock_operation(name=op_name, done=False)
        
        # 3. Polled operation - done, success
        mock_video_resource = MagicMock(spec=genai_types.VideoResource) # Structure for first_video.video
        mock_video_resource.name = "files/generated-video-file-id"
        
        mock_generated_video_entry = MagicMock(spec=genai_types.GeneratedVideo) # Structure for generated_videos list items
        mock_generated_video_entry.video = mock_video_resource # Assign the resource here

        final_op_success = _create_mock_operation(name=op_name, done=True, response_videos=[mock_generated_video_entry])
        
        mock_sync_client.operations.get.side_effect = [polling_op_not_done, final_op_success]

        # 4. Mock files.get
        mock_file_meta = MagicMock(spec=genai_types.File)
        mock_file_meta.mime_type = "video/mp4"
        mock_file_meta.name = "files/generated-video-file-id"
        mock_sync_client.files.get.return_value = mock_file_meta

        # 5. Mock files.download
        mock_download_response = MagicMock()
        mock_download_response.content = b"mock_video_content"
        mock_sync_client.files.download.return_value = mock_download_response

        system_prompt = "System prompt"
        user_prompt = "Make a video of a cat"
        image_data = b"fake_image_bytes"

        video_bytes, mime_type = await generate_video_with_veo(system_prompt, user_prompt, image_data)

        self.assertEqual(video_bytes, b"mock_video_content")
        self.assertEqual(mime_type, "video/mp4")

        MockGeminiClient.assert_called_once_with(api_key=GEMINI_API_KEY)
        mock_detect_mime.assert_called_once_with(image_data)
        
        # Check generate_videos call
        mock_sync_client.models.generate_videos.assert_called_once()
        call_args = mock_sync_client.models.generate_videos.call_args
        self.assertEqual(call_args.kwargs['model'], GEMINI_VIDEO_MODEL)
        self.assertEqual(call_args.kwargs['prompt'], f"{system_prompt}\n\n{user_prompt}")
        self.assertIsNotNone(call_args.kwargs['image'])
        self.assertEqual(call_args.kwargs['image'].data, image_data)
        self.assertEqual(call_args.kwargs['image'].mime_type, "image/jpeg")
        self.assertIsInstance(call_args.kwargs['config'], genai_types.GenerateVideosConfig)
        self.assertEqual(call_args.kwargs['safety_settings'], _safety_settings)

        # Check operations.get calls
        self.assertEqual(mock_sync_client.operations.get.call_count, 2)
        mock_sync_client.operations.get.assert_any_call(name=op_name)
        
        # Check sleep calls (called once between the two operations.get)
        mock_sleep.assert_called_once_with(20)

        mock_sync_client.files.get.assert_called_once_with(name="files/generated-video-file-id")
        mock_sync_client.files.download.assert_called_once_with(name="files/generated-video-file-id")


    @patch('bot.llm.time.sleep', return_value=None)
    @patch('bot.llm.detect_mime_type') # Not called, but patch to avoid interference
    @patch('bot.llm.genai.Client')
    async def test_generate_video_success_prompt_only(self, MockGeminiClient, mock_detect_mime, mock_sleep):
        mock_sync_client = MockGeminiClient.return_value
        op_name = "operations/prompt-op-456"
        initial_op = _create_mock_operation(name=op_name, done=False)
        mock_sync_client.models.generate_videos.return_value = initial_op

        mock_video_resource = MagicMock(spec=genai_types.VideoResource, name="files/prompt-video-id")
        mock_generated_video_entry = MagicMock(spec=genai_types.GeneratedVideo, video=mock_video_resource)
        final_op_success = _create_mock_operation(name=op_name, done=True, response_videos=[mock_generated_video_entry])
        mock_sync_client.operations.get.return_value = final_op_success # Only one poll needed

        mock_file_meta = MagicMock(spec=genai_types.File, mime_type="video/webm", name="files/prompt-video-id")
        mock_sync_client.files.get.return_value = mock_file_meta
        mock_download_response = MagicMock(content=b"prompt_video_data")
        mock_sync_client.files.download.return_value = mock_download_response

        system_prompt = "Sys"
        user_prompt = "Video please"
        video_bytes, mime_type = await generate_video_with_veo(system_prompt, user_prompt, image_data=None)

        self.assertEqual(video_bytes, b"prompt_video_data")
        self.assertEqual(mime_type, "video/webm")
        mock_detect_mime.assert_not_called()
        call_args = mock_sync_client.models.generate_videos.call_args
        self.assertIsNone(call_args.kwargs['image']) # Image should be None
        mock_sleep.assert_called_once_with(20)


    @patch('bot.llm.time.sleep', return_value=None)
    @patch('bot.llm.genai.Client')
    async def test_generate_video_operation_ok_no_videos(self, MockGeminiClient, mock_sleep):
        mock_sync_client = MockGeminiClient.return_value
        op_name = "operations/no-video-op-789"
        initial_op = _create_mock_operation(name=op_name, done=False)
        mock_sync_client.models.generate_videos.return_value = initial_op
        
        # Operation completes successfully but response has no videos
        final_op_no_video = _create_mock_operation(name=op_name, done=True, response_videos=[])
        mock_sync_client.operations.get.return_value = final_op_no_video

        video_bytes, mime_type = await generate_video_with_veo("S", "U", image_data=None)

        self.assertIsNone(video_bytes)
        self.assertIsNone(mime_type)
        mock_sync_client.files.get.assert_not_called()
        mock_sync_client.files.download.assert_not_called()
        # Check logger warning?
        # with self.assertLogs('bot.llm', level='WARNING') as cm:
        #     await generate_video_with_veo("S", "U", image_data=None)
        # self.assertIn("Video generation operation completed but no videos found", cm.output[0])


    @patch('bot.llm.genai.Client')
    async def test_generate_video_generate_videos_raises_exception(self, MockGeminiClient):
        mock_sync_client = MockGeminiClient.return_value
        mock_sync_client.models.generate_videos.side_effect = Exception("Generate videos API call failed")

        # Patch logger to check for error logging
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

        # Operation completes with an error
        final_op_with_error = _create_mock_operation(name=op_name, done=True, error_code=3) # 3 is a common gRPC error code for INVALID_ARGUMENT
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
        
        # Initial and all polled operations are not done
        persistent_not_done_op = _create_mock_operation(name=op_name, done=False)
        mock_sync_client.models.generate_videos.return_value = persistent_not_done_op
        mock_sync_client.operations.get.return_value = persistent_not_done_op

        # Max polling attempts is 30 in the code
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
    async def test_generate_video_files_get_raises_exception(self, MockGeminiClient, mock_sleep):
        mock_sync_client = MockGeminiClient.return_value
        op_name = "operations/files-get-fail-333"
        initial_op = _create_mock_operation(name=op_name, done=False)
        mock_sync_client.models.generate_videos.return_value = initial_op

        mock_video_resource = MagicMock(spec=genai_types.VideoResource, name="files/video-id-for-files-get-fail")
        mock_generated_video_entry = MagicMock(spec=genai_types.GeneratedVideo, video=mock_video_resource)
        final_op_success = _create_mock_operation(name=op_name, done=True, response_videos=[mock_generated_video_entry])
        mock_sync_client.operations.get.return_value = final_op_success
        
        mock_sync_client.files.get.side_effect = Exception("files.get API call failed")

        with patch('bot.llm.logger.error') as mock_logger_error:
            video_bytes, mime_type = await generate_video_with_veo("S", "U", image_data=None)
            self.assertIsNone(video_bytes)
            self.assertIsNone(mime_type)
            mock_logger_error.assert_any_call(
                "Exception during VEO video generation or polling: files.get API call failed", exc_info=True
            )

    @patch('bot.llm.time.sleep', return_value=None)
    @patch('bot.llm.genai.Client')
    async def test_generate_video_files_download_raises_exception(self, MockGeminiClient, mock_sleep):
        mock_sync_client = MockGeminiClient.return_value
        op_name = "operations/files-download-fail-444"
        initial_op = _create_mock_operation(name=op_name, done=False)
        mock_sync_client.models.generate_videos.return_value = initial_op

        mock_video_resource = MagicMock(spec=genai_types.VideoResource, name="files/video-id-for-files-download-fail")
        mock_generated_video_entry = MagicMock(spec=genai_types.GeneratedVideo, video=mock_video_resource)
        final_op_success = _create_mock_operation(name=op_name, done=True, response_videos=[mock_generated_video_entry])
        mock_sync_client.operations.get.return_value = final_op_success

        mock_file_meta = MagicMock(spec=genai_types.File, mime_type="video/mp4", name="files/video-id-for-files-download-fail")
        mock_sync_client.files.get.return_value = mock_file_meta
        
        mock_sync_client.files.download.side_effect = Exception("files.download API call failed")

        with patch('bot.llm.logger.error') as mock_logger_error:
            video_bytes, mime_type = await generate_video_with_veo("S", "U", image_data=None)
            self.assertIsNone(video_bytes)
            self.assertIsNone(mime_type)
            mock_logger_error.assert_any_call(
                "Exception during VEO video generation or polling: files.download API call failed", exc_info=True
            )

    @patch('bot.llm.time.sleep', return_value=None)
    @patch('bot.llm.detect_mime_type', return_value="application/pdf") # Invalid image MIME
    @patch('bot.llm.genai.Client')
    async def test_generate_video_invalid_image_mime_logs_error_and_skips_image(self, MockGeminiClient, mock_detect_mime, mock_sleep):
        mock_sync_client = MockGeminiClient.return_value
        op_name = "operations/invalid-mime-op-555"
        # Mock successful video generation without image
        initial_op = _create_mock_operation(name=op_name, done=False)
        mock_sync_client.models.generate_videos.return_value = initial_op

        mock_video_resource = MagicMock(spec=genai_types.VideoResource, name="files/video-no-image-id")
        mock_generated_video_entry = MagicMock(spec=genai_types.GeneratedVideo, video=mock_video_resource)
        final_op_success = _create_mock_operation(name=op_name, done=True, response_videos=[mock_generated_video_entry])
        mock_sync_client.operations.get.return_value = final_op_success
        
        mock_file_meta = MagicMock(spec=genai_types.File, mime_type="video/mp4", name="files/video-no-image-id")
        mock_sync_client.files.get.return_value = mock_file_meta
        mock_download_response = MagicMock(content=b"video_content_no_image")
        mock_sync_client.files.download.return_value = mock_download_response

        image_data = b"fake_pdf_bytes"
        with patch('bot.llm.logger.error') as mock_logger_error:
            video_bytes, mime_type = await generate_video_with_veo("S", "U", image_data=image_data)

        self.assertEqual(video_bytes, b"video_content_no_image")
        self.assertEqual(mime_type, "video/mp4")
        
        mock_detect_mime.assert_called_once_with(image_data)
        mock_logger_error.assert_any_call("Invalid MIME type for image_data: application/pdf. Skipping image.")
        
        # Verify that the 'image' argument to generate_videos was None
        call_args = mock_sync_client.models.generate_videos.call_args
        self.assertIsNone(call_args.kwargs['image'])

if __name__ == '__main__':
    unittest.main()
