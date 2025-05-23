import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from io import BytesIO

from telegram import Update, User, Message, Chat, PhotoSize, File
from telegram.ext import ContextTypes

from bot.handlers import vid_handler
# Ensure these can be imported. If not, the test environment needs adjustment.
# For this task, we assume they are importable.
# from bot.llm import generate_video_with_veo, download_media 

class TestVidHandler(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.update = MagicMock(spec=Update)
        self.context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)

        self.update.effective_chat = MagicMock(spec=Chat)
        self.update.effective_chat.id = 12345
        self.update.effective_user = MagicMock(spec=User)
        self.update.effective_user.id = 100
        self.update.effective_user.full_name = "Test User"
        
        self.update.effective_message = MagicMock(spec=Message)
        self.update.effective_message.message_id = 1
        self.update.effective_message.reply_to_message = None
        self.update.effective_message.text = "/vid test prompt"
        self.update.effective_message.caption = None
        self.update.effective_message.photo = None
        self.update.effective_message.video = None 

        # Mock bot methods
        self.context.bot = MagicMock() # Ensure bot object exists
        self.context.bot.get_file = AsyncMock(spec=File)
        self.context.bot.send_video = AsyncMock() # Though reply_video is used more often

        # Mock message methods
        self.update.effective_message.reply_text = AsyncMock()
        self.update.effective_message.reply_video = AsyncMock()
        self.update.effective_message.delete = AsyncMock() # For deleting "processing" message
        
        # Mock user_rate_limits from bot.handlers
        # Initialize with an empty dict for each test
        self.user_rate_limits_patch = patch('bot.handlers.user_rate_limits', {})
        self.mock_user_rate_limits = self.user_rate_limits_patch.start()
        
        # Mock RATE_LIMIT_SECONDS from bot.config, used by is_rate_limited in handlers
        self.rate_limit_seconds_patch = patch('bot.handlers.RATE_LIMIT_SECONDS', 5) # Example value
        self.mock_rate_limit_seconds = self.rate_limit_seconds_patch.start()


    async def asyncTearDown(self):
        self.user_rate_limits_patch.stop()
        self.rate_limit_seconds_patch.stop()

    @patch('bot.handlers.generate_video_with_veo', new_callable=AsyncMock)
    @patch('bot.handlers.download_media', new_callable=AsyncMock)
    async def test_vid_handler_prompt_only_success(self, mock_download_media, mock_generate_video):
        # Setup mock return values
        mock_generate_video.return_value = (b"videodata", "video/mp4")
        # Simulate a "processing" message that can be deleted
        self.update.effective_message.reply_text.return_value = self.update.effective_message

        self.update.effective_message.text = "/vid test prompt"

        await vid_handler(self.update, self.context)

        # Assert generate_video_with_veo was called correctly
        mock_generate_video.assert_called_once_with(
            system_prompt="You are a helpful video generation assistant.",
            user_prompt="test prompt",
            image_data=None
        )
        # Assert reply_video was called
        self.update.effective_message.reply_video.assert_called_once()
        args, kwargs = self.update.effective_message.reply_video.call_args
        self.assertIsInstance(kwargs['video'], BytesIO)
        self.assertEqual(kwargs['video'].read(), b"videodata")
        self.assertEqual(kwargs['caption'], "Here's your generated video!")
        
        # Assert download_media was not called
        mock_download_media.assert_not_called()
        # Assert the processing message was deleted
        self.update.effective_message.delete.assert_called_once()


    @patch('bot.handlers.generate_video_with_veo', new_callable=AsyncMock)
    @patch('bot.handlers.download_media', new_callable=AsyncMock)
    async def test_vid_handler_prompt_and_image_success(self, mock_download_media, mock_generate_video):
        self.update.effective_message.text = "/vid image prompt"
        
        # Mock replied message with photo
        replied_message = MagicMock(spec=Message)
        mock_photo = MagicMock(spec=PhotoSize)
        mock_photo.file_id = "photofileid"
        replied_message.photo = [mock_photo] 
        replied_message.caption = None
        self.update.effective_message.reply_to_message = replied_message
        
        # Mock file download process
        mock_file_obj = MagicMock(spec=File) # Renamed to avoid conflict
        mock_file_obj.file_path = "http://example.com/image.jpg"
        self.context.bot.get_file.return_value = mock_file_obj
        
        mock_download_media.return_value = b"imagedata"
        mock_generate_video.return_value = (b"videofromimage", "video/mp4")
        self.update.effective_message.reply_text.return_value = self.update.effective_message # For processing message

        await vid_handler(self.update, self.context)

        mock_download_media.assert_called_once_with("http://example.com/image.jpg")
        mock_generate_video.assert_called_once_with(
            system_prompt="You are a helpful video generation assistant.",
            user_prompt="image prompt",
            image_data=b"imagedata"
        )
        self.update.effective_message.reply_video.assert_called_once()
        args, kwargs = self.update.effective_message.reply_video.call_args
        self.assertEqual(kwargs['video'].read(), b"videofromimage")
        self.update.effective_message.delete.assert_called_once()


    @patch('bot.handlers.generate_video_with_veo', new_callable=AsyncMock)
    async def test_vid_handler_generation_fails(self, mock_generate_video):
        mock_generate_video.return_value = (None, None)
        self.update.effective_message.text = "/vid fail prompt"
        # Simulate a "processing" message that can be edited
        processing_msg_mock = AsyncMock()
        self.update.effective_message.reply_text.return_value = processing_msg_mock

        await vid_handler(self.update, self.context)

        mock_generate_video.assert_called_once()
        # Check that the "processing" message was edited with the failure message
        processing_msg_mock.edit_text.assert_called_with(
             "Sorry, I couldn't generate the video. Please try a different prompt or image. The model might have limitations or be unavailable."
        )
        self.update.effective_message.reply_video.assert_not_called()

    async def test_vid_handler_no_prompt_no_image(self):
        self.update.effective_message.text = "/vid"
        self.update.effective_message.reply_to_message = None

        await vid_handler(self.update, self.context)
        self.update.effective_message.reply_text.assert_called_with(
            "Please provide a prompt for the video, or reply to an image with a prompt (or use image caption as prompt).\n"
            "Usage: /vid [text prompt]\n"
            "Or reply to an image: /vid [optional text prompt]"
        )

    @patch('bot.handlers.time.time') 
    async def test_vid_handler_rate_limited(self, mock_time_time):
        user_id = self.update.effective_user.id
        
        # Setup time to simulate recent request
        # RATE_LIMIT_SECONDS is patched to 5 for this test
        self.mock_user_rate_limits[user_id] = 999.0 # A time in the past
        mock_time_time.return_value = 1000.0 # Current time
        
        # First call (advances time for user_rate_limits)
        # To correctly test is_rate_limited, we need to ensure it's called.
        # The issue is the global user_rate_limits dict.
        # We'll set it up as if a call just happened.
        self.mock_user_rate_limits[user_id] = mock_time_time.return_value - 2 # 2 seconds ago

        await vid_handler(self.update, self.context)
        
        self.update.effective_message.reply_text.assert_called_with(
            "You're sending commands too quickly. Please wait a moment before trying again."
        )

    @patch('bot.handlers.download_media', new_callable=AsyncMock)
    async def test_vid_handler_download_fails(self, mock_download_media):
        self.update.effective_message.text = "/vid download fail"
        
        replied_message = MagicMock(spec=Message)
        mock_photo = MagicMock(spec=PhotoSize)
        mock_photo.file_id = "photofileid"
        replied_message.photo = [mock_photo]
        replied_message.caption = None
        self.update.effective_message.reply_to_message = replied_message
        
        mock_file_obj = MagicMock(spec=File) # Renamed
        mock_file_obj.file_path = "http://example.com/image.jpg"
        self.context.bot.get_file.return_value = mock_file_obj
        
        mock_download_media.return_value = None # Simulate download failure

        await vid_handler(self.update, self.context)
        # This assertion needs to match the actual error message in vid_handler
        self.update.effective_message.reply_text.assert_called_with("Error downloading image. Please try again.")
        mock_download_media.assert_called_once()

    @patch('bot.handlers.generate_video_with_veo', new_callable=AsyncMock)
    async def test_vid_handler_telegram_send_error(self, mock_generate_video):
        mock_generate_video.return_value = (b"videodata", "video/mp4")
        self.update.effective_message.text = "/vid send error"
        self.update.effective_message.reply_video.side_effect = Exception("Telegram send failed")
        
        # Simulate a "processing" message that can be edited
        processing_msg_mock = AsyncMock()
        self.update.effective_message.reply_text.return_value = processing_msg_mock


        await vid_handler(self.update, self.context)
        
        # Check that the "processing" message was edited with the failure message
        processing_msg_mock.edit_text.assert_called_with(
            "Sorry, I generated the video but couldn't send it via Telegram. It might be too large or in an unsupported format."
        )

if __name__ == '__main__':
    unittest.main()
