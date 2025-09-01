"""Tests for the whitelist functionality."""

import os
import tempfile
import pytest
from unittest.mock import patch, mock_open

from bot.handlers import is_user_whitelisted, is_chat_whitelisted, is_access_allowed


class TestWhitelist:
    """Test cases for whitelist functionality."""

    def test_whitelist_file_not_exists(self):
        """Test that all users are allowed when whitelist file doesn't exist."""
        with patch('os.path.exists', return_value=False):
            assert is_user_whitelisted(123456789) is True

    def test_whitelist_file_exists_user_allowed(self):
        """Test that whitelisted users are allowed."""
        mock_file_content = "123456789\n987654321\n"
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=mock_file_content)):
                assert is_user_whitelisted(123456789) is True
                assert is_user_whitelisted(987654321) is True

    def test_whitelist_file_exists_user_not_allowed(self):
        """Test that non-whitelisted users are denied."""
        mock_file_content = "123456789\n987654321\n"
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=mock_file_content)):
                assert is_user_whitelisted(555555555) is False

    def test_whitelist_file_with_comments_and_empty_lines(self):
        """Test that comments and empty lines are ignored."""
        mock_file_content = "# This is a comment\n\n123456789\n  \n987654321\n# Another comment"
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=mock_file_content)):
                assert is_user_whitelisted(123456789) is True
                assert is_user_whitelisted(987654321) is True
                assert is_user_whitelisted(555555555) is False

    def test_whitelist_file_read_error(self):
        """Test that read errors result in access denied for security."""
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', side_effect=Exception("File read error")):
                assert is_user_whitelisted(123456789) is False

    def test_whitelist_with_string_user_id(self):
        """Test that string user IDs are handled correctly."""
        mock_file_content = "123456789\n987654321\n"
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=mock_file_content)):
                # The function converts user_id to string for comparison
                assert is_user_whitelisted(123456789) is True
                assert is_user_whitelisted("123456789") is False  # Should be int


class TestChatWhitelist:
    """Test cases for chat whitelist functionality."""

    def test_chat_whitelist_file_not_exists(self):
        """Test that all chats are allowed when whitelist file doesn't exist."""
        with patch('os.path.exists', return_value=False):
            assert is_chat_whitelisted(-1001234567890) is True

    def test_chat_whitelist_file_exists_chat_allowed(self):
        """Test that whitelisted chats are allowed."""
        mock_file_content = "-1001234567890\n-1009876543210\n"
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=mock_file_content)):
                assert is_chat_whitelisted(-1001234567890) is True
                assert is_chat_whitelisted(-1009876543210) is True

    def test_chat_whitelist_file_exists_chat_not_allowed(self):
        """Test that non-whitelisted chats are denied."""
        mock_file_content = "-1001234567890\n-1009876543210\n"
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=mock_file_content)):
                assert is_chat_whitelisted(-1005555555555) is False

    def test_chat_whitelist_with_mixed_ids(self):
        """Test that chat whitelist works with mixed user and chat IDs."""
        mock_file_content = "123456789\n-1001234567890\n987654321\n"
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=mock_file_content)):
                # Chat IDs should be found
                assert is_chat_whitelisted(-1001234567890) is True
                # User IDs should not be found in chat whitelist
                assert is_chat_whitelisted(123456789) is False


class TestAccessControl:
    """Test cases for combined access control functionality."""

    def test_access_allowed_user_whitelisted(self):
        """Test that access is allowed when user is whitelisted."""
        with patch('bot.handlers.is_user_whitelisted', return_value=True):
            with patch('bot.handlers.is_chat_whitelisted', return_value=False):
                assert is_access_allowed(123456789, -1001234567890) is True

    def test_access_allowed_chat_whitelisted(self):
        """Test that access is allowed when chat is whitelisted."""
        with patch('bot.handlers.is_user_whitelisted', return_value=False):
            with patch('bot.handlers.is_chat_whitelisted', return_value=True):
                assert is_access_allowed(123456789, -1001234567890) is True

    def test_access_allowed_both_whitelisted(self):
        """Test that access is allowed when both user and chat are whitelisted."""
        with patch('bot.handlers.is_user_whitelisted', return_value=True):
            with patch('bot.handlers.is_chat_whitelisted', return_value=True):
                assert is_access_allowed(123456789, -1001234567890) is True

    def test_access_denied_neither_whitelisted(self):
        """Test that access is denied when neither user nor chat is whitelisted."""
        with patch('bot.handlers.is_user_whitelisted', return_value=False):
            with patch('bot.handlers.is_chat_whitelisted', return_value=False):
                assert is_access_allowed(123456789, -1001234567890) is False
