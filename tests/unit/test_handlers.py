import unittest
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

from telegram import Chat, File, Message, PhotoSize, Update, User
from telegram.ext import ContextTypes

from bot.config import PAINTME_SYSTEM_PROMPT, PROFILEME_SYSTEM_PROMPT
from bot.db.models import Message as DBMessage  # Alias to avoid conflict with telegram.Message
from bot.handlers import img_handler, paintme_handler, profileme_handler, vid_handler

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


class TestImgHandler(unittest.IsolatedAsyncioTestCase):

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
        self.update.effective_message.text = "/img test prompt"
        self.update.effective_message.photo = None
        
        # Mock processing_message that is returned by reply_text
        self.processing_message = AsyncMock(spec=Message)
        self.update.effective_message.reply_text = AsyncMock(return_value=self.processing_message)

        # Mock bot methods
        self.context.bot = MagicMock()
        self.context.bot.send_media_group = AsyncMock()
        # Note: edit_media is a method of the Message object (processing_message)
        
        # Mock user_rate_limits and RATE_LIMIT_SECONDS from bot.handlers
        self.user_rate_limits_patch = patch('bot.handlers.user_rate_limits', {})
        self.mock_user_rate_limits = self.user_rate_limits_patch.start()
        
        self.rate_limit_seconds_patch = patch('bot.handlers.RATE_LIMIT_SECONDS', 5)
        self.mock_rate_limit_seconds = self.rate_limit_seconds_patch.start()

        # Mock queue_message_insert
        self.queue_message_insert_patch = patch('bot.handlers.queue_message_insert', AsyncMock())
        self.mock_queue_message_insert = self.queue_message_insert_patch.start()
        
        # Mock langid.classify
        self.langid_classify_patch = patch('bot.handlers.langid.classify', return_value=('en', 0.9))
        self.mock_langid_classify = self.langid_classify_patch.start()


    async def asyncTearDown(self):
        self.user_rate_limits_patch.stop()
        self.rate_limit_seconds_patch.stop()
        self.queue_message_insert_patch.stop()
        self.langid_classify_patch.stop()

    # --- Test Vertex AI Path (USE_VERTEX_IMAGE = True) ---
    @patch('bot.handlers.USE_VERTEX_IMAGE', True)
    @patch('bot.handlers.generate_image_with_vertex', new_callable=AsyncMock)
    async def test_img_handler_vertex_multiple_images(self, mock_generate_vertex):
        self.update.effective_message.text = "/img vertex multi prompt"
        mock_generate_vertex.return_value = [b'img1', b'img2', b'img3', b'img4']

        await img_handler(self.update, self.context)

        mock_generate_vertex.assert_called_once_with(prompt="vertex multi prompt")
        self.context.bot.send_media_group.assert_called_once()
        
        # Check media group content
        media_group_call_args = self.context.bot.send_media_group.call_args
        self.assertEqual(media_group_call_args.kwargs['chat_id'], self.update.effective_chat.id)
        media_items = media_group_call_args.kwargs['media']
        self.assertEqual(len(media_items), 4)
        self.assertIsNotNone(media_items[0].caption) # First image has caption
        self.assertEqual(media_items[0].caption, "Images generated with Vertex AI, hope you like them!")
        for i in range(1, 4):
            self.assertIsNone(media_items[i].caption) # Others don't

        self.processing_message.delete.assert_called_once()
        self.processing_message.edit_media.assert_not_called()
        self.processing_message.edit_text.assert_not_called() # Should not be called on success with multi-image

    @patch('bot.handlers.USE_VERTEX_IMAGE', True)
    @patch('bot.handlers.generate_image_with_vertex', new_callable=AsyncMock)
    async def test_img_handler_vertex_single_image(self, mock_generate_vertex):
        self.update.effective_message.text = "/img vertex single prompt"
        mock_generate_vertex.return_value = [b'img1']

        await img_handler(self.update, self.context)

        mock_generate_vertex.assert_called_once_with(prompt="vertex single prompt")
        self.processing_message.edit_media.assert_called_once()
        
        edit_media_call_args = self.processing_message.edit_media.call_args
        media_arg = edit_media_call_args.kwargs['media']
        self.assertEqual(media_arg.caption, "Image generated with Vertex AI, hope you like it.")
        
        self.context.bot.send_media_group.assert_not_called()
        self.processing_message.delete.assert_not_called()

    @patch('bot.handlers.USE_VERTEX_IMAGE', True)
    @patch('bot.handlers.generate_image_with_vertex', new_callable=AsyncMock)
    async def test_img_handler_vertex_no_images_returned(self, mock_generate_vertex):
        self.update.effective_message.text = "/img vertex no images"
        mock_generate_vertex.return_value = []

        await img_handler(self.update, self.context)

        mock_generate_vertex.assert_called_once_with(prompt="vertex no images")
        self.processing_message.edit_text.assert_called_once_with(
            "Sorry, I couldn't generate images using Vertex AI. Please try again."
        )
        self.processing_message.edit_media.assert_not_called()
        self.context.bot.send_media_group.assert_not_called()

    # --- Test Gemini Path (USE_VERTEX_IMAGE = False) ---
    @patch('bot.handlers.USE_VERTEX_IMAGE', False)
    @patch('bot.handlers.generate_image_with_gemini', new_callable=AsyncMock)
    async def test_img_handler_gemini_success(self, mock_generate_gemini):
        self.update.effective_message.text = "/img gemini good prompt"
        mock_generate_gemini.return_value = b'gemini_img_bytes'

        await img_handler(self.update, self.context)

        mock_generate_gemini.assert_called_once_with(
            system_prompt="Generate an image based on the description.", # This is hardcoded in handler for Gemini
            prompt="gemini good prompt",
            input_image_url=None 
        )
        self.processing_message.edit_media.assert_called_once()
        edit_media_call_args = self.processing_message.edit_media.call_args
        media_arg = edit_media_call_args.kwargs['media']
        self.assertEqual(media_arg.caption, "Image generated with Gemini, hope you like it.")

    @patch('bot.handlers.USE_VERTEX_IMAGE', False)
    @patch('bot.handlers.generate_image_with_gemini', new_callable=AsyncMock)
    async def test_img_handler_gemini_failure_no_image_url(self, mock_generate_gemini):
        self.update.effective_message.text = "/img gemini bad prompt"
        mock_generate_gemini.return_value = None # Simulate generation failure

        await img_handler(self.update, self.context)

        mock_generate_gemini.assert_called_once_with(
            system_prompt="Generate an image based on the description.",
            prompt="gemini bad prompt",
            input_image_url=None
        )
        self.processing_message.edit_text.assert_called_once_with(
            "I couldn't generate an image based on your request. The Gemini model may have limitations with "
            "image generation capabilities. Please try:\n"
            "1. Using a simpler prompt\n"
            "2. Providing more specific details\n"
            "3. Try again later as model capabilities continue to improve"
        )

    @patch('bot.handlers.USE_VERTEX_IMAGE', False)
    @patch('bot.handlers.generate_image_with_gemini', new_callable=AsyncMock)
    @patch('bot.handlers.download_media', new_callable=AsyncMock)  # Not used here but good to have if logic changes
    async def test_img_handler_gemini_failure_with_image_url(self, mock_download_media, mock_generate_gemini):
        self.update.effective_message.text = "/img gemini edit bad"
        
        # Mock replied message with photo
        replied_message = MagicMock(spec=Message)
        mock_photo = MagicMock(spec=PhotoSize, file_id="photofailid")
        replied_message.photo = [mock_photo] 
        self.update.effective_message.reply_to_message = replied_message
        
        # Mock file object and path for image_url
        mock_file_obj = MagicMock(spec=File, file_path="http://example.com/image_to_edit.jpg")
        self.context.bot.get_file = AsyncMock(return_value=mock_file_obj)
        
        mock_generate_gemini.return_value = None # Simulate generation failure

        await img_handler(self.update, self.context)

        self.context.bot.get_file.assert_called_once_with("photofailid")
        mock_generate_gemini.assert_called_once_with(
            system_prompt="Generate an image based on the description.",
            prompt="gemini edit bad",
            input_image_url="http://example.com/image_to_edit.jpg" 
        )
        self.processing_message.edit_text.assert_called_once_with(
             "I couldn't edit the image according to your request. The Gemini model may have limitations "
             "with image editing capabilities. Please try:\n"
             "1. Using a simpler edit description\n"
             "2. Providing more specific details\n"
             "3. Try a different type of edit or try again later"
        )

    # --- Test Prompt Extraction and Edge Cases ---
    async def test_img_handler_no_prompt(self):
        self.update.effective_message.text = "/img" # No prompt
        
        # Redefine reply_text on original message for this test, not processing_message
        original_message_reply_text = AsyncMock()
        self.update.effective_message.reply_text = original_message_reply_text
        
        await img_handler(self.update, self.context)
        
        original_message_reply_text.assert_called_once_with(
             "Please provide a description of the image you want to generate or edit. "
             "For example: /img a cat playing piano"
        )
        # Ensure processing_message methods were not called as it shouldn't get that far
        self.processing_message.edit_media.assert_not_called()
        self.processing_message.edit_text.assert_not_called()
        self.context.bot.send_media_group.assert_not_called()

    @patch('bot.handlers.is_rate_limited', return_value=True)
    async def test_img_handler_rate_limited(self, mock_is_rate_limited):
        self.update.effective_message.text = "/img some prompt"
        
        # Redefine reply_text on original message for this test
        original_message_reply_text = AsyncMock()
        self.update.effective_message.reply_text = original_message_reply_text

        await img_handler(self.update, self.context)

        mock_is_rate_limited.assert_called_once_with(self.update.effective_user.id)
        original_message_reply_text.assert_called_once_with(
            "You're sending commands too quickly. Please wait a moment before trying again."
        )
        # Ensure no processing message was created or interacted with
        self.processing_message.edit_media.assert_not_called()
        self.processing_message.edit_text.assert_not_called()


# Ensure InputMediaPhoto is imported for context if not already.
# from telegram import InputMediaPhoto # Already imported in handlers.py, so available in its namespace.
# No, tests need their own imports if they reference it directly, e.g. for type checking or constructing.
# However, here we are checking `InputMediaPhoto` instances created *within* the handler.
# The mocks for `generate_image_with_vertex` and `generate_image_with_gemini` are at `bot.handlers.*`
# because that's where they are called from *within* `img_handler`.
# `USE_VERTEX_IMAGE` is also correctly patched at `bot.handlers.USE_VERTEX_IMAGE`.
# `BytesIO` is used within the handler, so it's fine as is, not directly asserted on type in tests.
# `queue_message_insert` and `langid.classify` are also dependencies of `img_handler`.
# Added mocks for them in `asyncSetUp`.

class TestProfileMeHandler(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.update = MagicMock(spec=Update)
        self.context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)

        self.update.effective_chat = MagicMock(spec=Chat)
        self.update.effective_chat.id = 12345
        self.update.effective_user = MagicMock(spec=User)
        self.update.effective_user.id = 100
        self.update.effective_user.full_name = "Test User Profile"
        
        self.update.effective_message = MagicMock(spec=Message)
        self.update.effective_message.message_id = 1
        self.update.effective_message.from_user = self.update.effective_user # Ensure from_user is set
        self.update.effective_message.text = "/profileme"

        self.processing_message = AsyncMock(spec=Message)
        self.update.effective_message.reply_text = AsyncMock(return_value=self.processing_message)

        # Patching dependencies
        self.user_rate_limits_patch = patch('bot.handlers.user_rate_limits', {}) # For general is_rate_limited
        self.mock_user_rate_limits = self.user_rate_limits_patch.start()
        
        self.rate_limit_seconds_patch = patch('bot.handlers.RATE_LIMIT_SECONDS', 5) # For general is_rate_limited
        self.mock_rate_limit_seconds = self.rate_limit_seconds_patch.start()

        # Specific patch for select_messages_by_user
        self.select_messages_by_user_patch = patch('bot.handlers.select_messages_by_user', new_callable=AsyncMock)
        self.mock_select_messages_by_user = self.select_messages_by_user_patch.start()
        
        # Patch USER_HISTORY_MESSAGE_COUNT for consistent testing
        self.user_history_count_patch = patch('bot.handlers.USER_HISTORY_MESSAGE_COUNT', 200) # Example value
        self.mock_user_history_count = self.user_history_count_patch.start()

        self.call_gemini_patch = patch('bot.handlers.call_gemini', new_callable=AsyncMock)
        self.mock_call_gemini = self.call_gemini_patch.start()

        self.send_response_patch = patch('bot.handlers.send_response', new_callable=AsyncMock)
        self.mock_send_response = self.send_response_patch.start()
        
        self.queue_message_insert_patch = patch('bot.handlers.queue_message_insert', AsyncMock())
        self.mock_queue_message_insert = self.queue_message_insert_patch.start()

    async def asyncTearDown(self):
        self.user_rate_limits_patch.stop()
        self.rate_limit_seconds_patch.stop()
        self.select_messages_by_user_patch.stop()
        self.user_history_count_patch.stop()
        self.call_gemini_patch.stop()
        self.send_response_patch.stop()
        self.queue_message_insert_patch.stop()

    async def test_profileme_success(self):
        sample_messages = [
            DBMessage(user_id=100, username="Test User Profile", text="Hello world", date=MagicMock(), chat_id=12345, message_id=10),
            DBMessage(user_id=100, username="Test User Profile", text="This is a test", date=MagicMock(), chat_id=12345, message_id=11),
        ]
        self.mock_select_messages_by_user.return_value = sample_messages
        self.mock_call_gemini.return_value = "This is your generated profile."
        
        # Import USER_HISTORY_MESSAGE_COUNT here or access via self.mock_user_history_count if needed for assertion
        from bot.config import (
            USER_HISTORY_MESSAGE_COUNT as UHC_CONFIG,  # Using actual configured for assertion consistency
        )

        await profileme_handler(self.update, self.context)

        self.mock_select_messages_by_user.assert_called_once_with(
            chat_id=12345, 
            user_id=100, 
            limit=UHC_CONFIG # Assert with the patched or actual value
        )
        
        # Construct expected formatted_history
        expected_history = "Here is the user's recent chat history in this group:\n\n"
        for msg in sample_messages:
            expected_history += f"{msg.date.strftime('%Y-%m-%d %H:%M:%S')}: {msg.text}\n"
            
        self.mock_call_gemini.assert_called_once_with(
            system_prompt=PROFILEME_SYSTEM_PROMPT,
            user_content=expected_history,
            use_search_grounding=False
        )
        self.mock_send_response.assert_called_once_with(
            self.processing_message, "This is your generated profile.", "Your User Profile"
        )
        self.mock_queue_message_insert.assert_called_once() # Check if logging was called

    async def test_profileme_no_messages(self):
        self.mock_select_messages_by_user.return_value = [] # No messages for the user
        from bot.config import USER_HISTORY_MESSAGE_COUNT as UHC_CONFIG

        await profileme_handler(self.update, self.context)
        
        self.mock_select_messages_by_user.assert_called_once_with(
            chat_id=12345, user_id=100, limit=UHC_CONFIG
        )
        self.processing_message.edit_text.assert_called_once_with(
            "I don't have enough of your messages in this chat to generate a profile yet (need some from your recent history). Keep chatting!"
        )
        self.mock_call_gemini.assert_not_called()
        self.mock_send_response.assert_not_called()

    async def test_profileme_gemini_fails(self):
        sample_messages = [DBMessage(user_id=100, username="Test User Profile", text="Hello", date=MagicMock(), chat_id=12345, message_id=10)]
        self.mock_select_messages_by_user.return_value = sample_messages
        self.mock_call_gemini.return_value = None # Simulate Gemini failure

        await profileme_handler(self.update, self.context)

        self.processing_message.edit_text.assert_called_once_with(
            "I couldn't generate a profile at this time. Please try again later."
        )
        self.mock_send_response.assert_not_called()

    @patch('bot.handlers.is_rate_limited', return_value=True)
    async def test_profileme_rate_limited(self, mock_is_rate_limited_direct):
        # This test now directly mocks is_rate_limited for simplicity
        original_reply_text_mock = AsyncMock() # Mock reply_text on the original message
        self.update.effective_message.reply_text = original_reply_text_mock

        await profileme_handler(self.update, self.context)

        mock_is_rate_limited_direct.assert_called_once_with(self.update.effective_user.id)
        original_reply_text_mock.assert_called_with("Rate limit exceeded. Please try again later.")
        self.processing_message.edit_text.assert_not_called() # Ensure processing message not used


class TestPaintMeHandler(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.update = MagicMock(spec=Update)
        self.context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)

        self.update.effective_chat = MagicMock(spec=Chat)
        self.update.effective_chat.id = 12345
        self.update.effective_user = MagicMock(spec=User)
        self.update.effective_user.id = 200
        self.update.effective_user.full_name = "Painter User"
        
        self.update.effective_message = MagicMock(spec=Message)
        self.update.effective_message.message_id = 2
        self.update.effective_message.from_user = self.update.effective_user
        self.update.effective_message.text = "/paintme"
        self.update.effective_message.reply_photo = AsyncMock() # Important for sending the image

        self.processing_message = AsyncMock(spec=Message)
        self.update.effective_message.reply_text = AsyncMock(return_value=self.processing_message)

        # Patching dependencies
        self.user_rate_limits_patch = patch('bot.handlers.user_rate_limits', {}) # For general is_rate_limited
        self.mock_user_rate_limits = self.user_rate_limits_patch.start()
        
        self.rate_limit_seconds_patch = patch('bot.handlers.RATE_LIMIT_SECONDS', 5) # For general is_rate_limited
        self.mock_rate_limit_seconds = self.rate_limit_seconds_patch.start()

        # Specific patch for select_messages_by_user
        self.select_messages_by_user_patch = patch('bot.handlers.select_messages_by_user', new_callable=AsyncMock)
        self.mock_select_messages_by_user = self.select_messages_by_user_patch.start()

        # Patch USER_HISTORY_MESSAGE_COUNT for consistent testing
        self.user_history_count_patch = patch('bot.handlers.USER_HISTORY_MESSAGE_COUNT', 50) # Using 50 for paintme related tests for slicing
        self.mock_user_history_count = self.user_history_count_patch.start()

        self.call_gemini_patch = patch('bot.handlers.call_gemini', new_callable=AsyncMock)
        self.mock_call_gemini = self.call_gemini_patch.start()

        self.generate_image_gemini_patch = patch('bot.handlers.generate_image_with_gemini', new_callable=AsyncMock)
        self.mock_generate_image_gemini = self.generate_image_gemini_patch.start()
        
        self.generate_image_vertex_patch = patch('bot.handlers.generate_image_with_vertex', new_callable=AsyncMock)
        self.mock_generate_image_vertex = self.generate_image_vertex_patch.start()

        self.queue_message_insert_patch = patch('bot.handlers.queue_message_insert', AsyncMock())
        self.mock_queue_message_insert = self.queue_message_insert_patch.start()
        
        self.langid_classify_patch = patch('bot.handlers.langid.classify', return_value=('en', 0.9))
        self.mock_langid_classify = self.langid_classify_patch.start()


    async def asyncTearDown(self):
        self.user_rate_limits_patch.stop()
        self.rate_limit_seconds_patch.stop()
        self.select_messages_by_user_patch.stop()
        self.user_history_count_patch.stop()
        self.call_gemini_patch.stop()
        self.generate_image_gemini_patch.stop()
        self.generate_image_vertex_patch.stop()
        self.queue_message_insert_patch.stop()
        self.langid_classify_patch.stop()

    def _get_sample_messages(self, count=5):
        return [
            DBMessage(user_id=200, username="Painter User", text=f"Painting message {i}", date=MagicMock(), chat_id=12345, message_id=i+20)
            for i in range(count)
        ]

    @patch('bot.handlers.USE_VERTEX_IMAGE', False)  # Test Gemini Path
    async def test_paintme_success_gemini(self):
        # Test with USER_HISTORY_MESSAGE_COUNT returning exactly 50 messages
        # USER_HISTORY_MESSAGE_COUNT is patched to 50 in setUp
        self.mock_select_messages_by_user.return_value = self._get_sample_messages(count=50) 
        self.mock_call_gemini.return_value = "A beautiful sunset" # Mocked image prompt
        self.mock_generate_image_gemini.return_value = b"gemini_image_bytes"
        
        from bot.config import USER_HISTORY_MESSAGE_COUNT as UHC_CONFIG  # Get patched value

        await paintme_handler(self.update, self.context)

        self.mock_select_messages_by_user.assert_called_once_with(
            chat_id=12345, 
            user_id=200, 
            limit=UHC_CONFIG
        )
        
        # Verify formatted_history based on the (potentially sliced) 50 messages
        expected_formatted_history = "User's recent messages in the group chat:\n\n"
        for msg in self._get_sample_messages(count=50): # The slicing would result in these 50
             expected_formatted_history += f"- \"{msg.text}\"\n"
        
        self.mock_call_gemini.assert_called_once_with(
            system_prompt=PAINTME_SYSTEM_PROMPT,
            user_content=expected_formatted_history,
            use_search_grounding=False
        )
        self.mock_generate_image_gemini.assert_called_once_with(
            system_prompt="Generate an image based on the following description.",
            prompt="A beautiful sunset"
        )
        self.update.effective_message.reply_photo.assert_called_once()
        args, kwargs = self.update.effective_message.reply_photo.call_args
        self.assertIsInstance(kwargs['photo'], BytesIO)
        self.assertEqual(kwargs['photo'].read(), b"gemini_image_bytes")
        self.assertIn("A beautiful sunset", kwargs['caption'])
        self.processing_message.delete.assert_called_once()
        self.mock_queue_message_insert.assert_called_once()

    @patch('bot.handlers.USE_VERTEX_IMAGE', True)  # Test Vertex Path
    async def test_paintme_success_vertex(self):
        # Test with USER_HISTORY_MESSAGE_COUNT returning more than 50, e.g., 60
        # USER_HISTORY_MESSAGE_COUNT is patched to 50 in setUp for this class,
        # but handler logic fetches USER_HISTORY_MESSAGE_COUNT then slices to 50.
        # So, select_messages_by_user should be called with the *handler's* USER_HISTORY_MESSAGE_COUNT.
        # Let's simulate actual USER_HISTORY_MESSAGE_COUNT from config is higher for fetching, 
        # but the test's internal USER_HISTORY_MESSAGE_COUNT (for assertion) is 50.
        
        # For this test, let's assume USER_HISTORY_MESSAGE_COUNT (handler's import) is, say, 70,
        # and we return 60 messages. The handler will then slice it to 50.
        # The patch in setup for USER_HISTORY_MESSAGE_COUNT in handlers is 50.
        # So, select_messages_by_user will be called with limit=50.
        # If we want to test the slicing, we need select_messages_by_user to return > 50 messages
        # when called with a limit that allows it (e.g. the actual USER_HISTORY_MESSAGE_COUNT from config).
        # This test setup is getting a bit complex due to the two places USER_HISTORY_MESSAGE_COUNT is used.

        # Simplified: Assume USER_HISTORY_MESSAGE_COUNT (patched to 50 in setUp) is used for fetching.
        # If select_messages_by_user returns 50 messages, no slicing occurs.
        # If it returns 40, no slicing. If it returns 60 (mocked), it's sliced.
        # Let's mock it to return 60 messages, ensure only 50 are used for prompt.
        
        all_fetched_messages = self._get_sample_messages(count=60)
        # USER_HISTORY_MESSAGE_COUNT is patched to 50 in paintme_handler's scope for these tests.
        # So select_messages_by_user will be called with limit=50.
        # So we should mock its return to be 50 messages (or less).
        self.mock_select_messages_by_user.return_value = all_fetched_messages[:50]

        self.mock_call_gemini.return_value = "A serene lake"
        self.mock_generate_image_vertex.return_value = [b"vertex_image_bytes"]
        
        from bot.config import USER_HISTORY_MESSAGE_COUNT as UHC_CONFIG  # Get patched value (50)

        await paintme_handler(self.update, self.context)

        self.mock_select_messages_by_user.assert_called_once_with(
            chat_id=12345, user_id=200, limit=UHC_CONFIG
        )
        
        # Verify formatted_history based on the LAST 50 messages if more were hypothetically fetched and sliced.
        # Since USER_HISTORY_MESSAGE_COUNT is patched to 50, and select_messages_by_user returns 50,
        # the formatted history will be based on these 50.
        expected_formatted_history = "User's recent messages in the group chat:\n\n"
        for msg in all_fetched_messages[:50]: # Sliced to 50
             expected_formatted_history += f"- \"{msg.text}\"\n"

        self.mock_call_gemini.assert_called_once_with(
            system_prompt=PAINTME_SYSTEM_PROMPT,
            user_content=expected_formatted_history, # Should be based on the (potentially) sliced messages
            use_search_grounding=False
        )
        self.mock_generate_image_vertex.assert_called_once_with(prompt="A serene lake", number_of_images=1)
        self.update.effective_message.reply_photo.assert_called_once()
        args, kwargs = self.update.effective_message.reply_photo.call_args
        self.assertIsInstance(kwargs['photo'], BytesIO)
        self.assertEqual(kwargs['photo'].read(), b"vertex_image_bytes")
        self.assertIn("A serene lake", kwargs['caption'])
        self.processing_message.delete.assert_called_once()

    async def test_paintme_no_messages(self):
        # Test case: select_messages_by_user returns less than 5 messages
        self.mock_select_messages_by_user.return_value = self._get_sample_messages(count=3)
        from bot.config import USER_HISTORY_MESSAGE_COUNT as UHC_CONFIG

        await paintme_handler(self.update, self.context)
        
        self.mock_select_messages_by_user.assert_called_once_with(
            chat_id=12345, user_id=200, limit=UHC_CONFIG
        )
        self.processing_message.edit_text.assert_called_once_with(
            "I don't have enough of your messages (at least 5 recent ones) in this chat to paint a picture. Keep chatting!"
        )
        self.mock_call_gemini.assert_not_called()

    async def test_paintme_prompt_generation_fails(self):
        self.mock_select_messages_by_user.return_value = self._get_sample_messages(count=10) # Enough messages
        self.mock_call_gemini.return_value = None # Failed to generate prompt

        await paintme_handler(self.update, self.context)
        
        self.processing_message.edit_text.assert_any_call( # First call is "Generated image prompt..."
            "I couldn't come up with an image idea for you at this time. Please try again later."
        )

    @patch('bot.handlers.USE_VERTEX_IMAGE', False)
    async def test_paintme_image_generation_fails_gemini(self):
        self.mock_select_messages_by_user.return_value = self._get_sample_messages(count=10)
        self.mock_call_gemini.return_value = "A test prompt"
        self.mock_generate_image_gemini.return_value = None # Gemini image gen fails

        await paintme_handler(self.update, self.context)

        self.processing_message.edit_text.assert_called_with(
            "Sorry, Gemini couldn't paint your picture. Please try again."
        )

    @patch('bot.handlers.USE_VERTEX_IMAGE', True)
    async def test_paintme_image_generation_fails_vertex(self):
        self.mock_select_messages_by_user.return_value = self._get_sample_messages(count=10)
        self.mock_call_gemini.return_value = "A test prompt"
        self.mock_generate_image_vertex.return_value = [] # Vertex image gen fails

        await paintme_handler(self.update, self.context)
        
        # Get the actual VERTEX_IMAGE_MODEL name for the message
        from bot.config import VERTEX_IMAGE_MODEL  # Import locally for this
        expected_message = f"Sorry, {VERTEX_IMAGE_MODEL} couldn't paint your picture. Please try again."
        self.processing_message.edit_text.assert_called_with(expected_message)

    @patch('bot.handlers.is_rate_limited', return_value=True)
    async def test_paintme_rate_limited(self, mock_is_rate_limited_direct):
        original_reply_text_mock = AsyncMock()
        self.update.effective_message.reply_text = original_reply_text_mock
        
        await paintme_handler(self.update, self.context)

        mock_is_rate_limited_direct.assert_called_once_with(self.update.effective_user.id)
        original_reply_text_mock.assert_called_with("Rate limit exceeded. Please try again later.")
        self.processing_message.edit_text.assert_not_called()

# Need to import ANY from unittest.mock for the paintme_handler test
# Import USER_HISTORY_MESSAGE_COUNT for assertions
