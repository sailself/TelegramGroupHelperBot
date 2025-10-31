"""Test Telegraph functionality for handling long messages."""

import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from bs4 import BeautifulSoup

from bot.handlers import create_telegraph_page, html_to_telegraph_nodes, markdown_to_telegraph_nodes


class TestTelegraphFunctionality(unittest.TestCase):
    """Test Telegraph functionality for handling long messages."""

    def test_markdown_to_telegraph_nodes(self):
        """Test conversion from markdown to Telegraph nodes."""
        # Test markdown content
        markdown_content = """# Test Heading
        
        This is a **bold text** and *italic text*.
        
        - List item 1
        - List item 2
        
        [A link](https://example.com)
        """
        
        # Convert to nodes
        nodes = markdown_to_telegraph_nodes(markdown_content)
        
        # Basic assertions to ensure conversion happened correctly
        self.assertIsInstance(nodes, list)
        self.assertTrue(len(nodes) > 0)
        
        # Look for specific elements
        found_heading = False
        found_list = False
        found_link = False
        
        for node in nodes:
            if isinstance(node, dict):
                if node.get('tag') == 'h1':
                    found_heading = True
                if node.get('tag') == 'ul':
                    found_list = True
                if node.get('tag') == 'a' and node.get('attrs', {}).get('href') == 'https://example.com':
                    found_link = True
        
        self.assertTrue(found_heading, "Failed to convert heading")
        self.assertTrue(found_list, "Failed to convert list")
        self.assertTrue(found_link, "Failed to convert link")

    def test_html_to_telegraph_nodes(self):
        """Test conversion from HTML to Telegraph nodes."""
        # Test HTML content
        html_content = """
        <h1>Test Heading</h1>
        <p>This is a <b>bold text</b> and <i>italic text</i>.</p>
        <ul>
            <li>List item 1</li>
            <li>List item 2</li>
        </ul>
        <p><a href="https://example.com">A link</a></p>
        """
        
        # Parse HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Convert to nodes
        nodes = html_to_telegraph_nodes(soup)
        
        # Basic assertions
        self.assertIsInstance(nodes, list)
        self.assertTrue(len(nodes) > 0)
        
        # Validate structure
        for node in nodes:
            if isinstance(node, dict) and node.get('tag') == 'a':
                self.assertIn('attrs', node)
                self.assertIn('href', node['attrs'])
                self.assertEqual(node['attrs']['href'], 'https://example.com')

    @pytest.mark.asyncio
    async def test_create_telegraph_page(self):
        """Test the creation of a Telegraph page."""
        # Mock the shared HTTP session
        with patch('bot.handlers.content.get_http_session', new_callable=AsyncMock) as mock_get_session:
            mock_session = AsyncMock()
            mock_get_session.return_value = mock_session

            mock_response = AsyncMock()
            mock_response.json = AsyncMock(return_value={
                'ok': True,
                'result': {'url': 'https://telegra.ph/Test-Page-01-01'}
            })
            mock_response.raise_for_status = MagicMock()

            mock_ctx = AsyncMock()
            mock_ctx.__aenter__.return_value = mock_response
            mock_ctx.__aexit__.return_value = False

            mock_session.post = AsyncMock(return_value=mock_ctx)

            result = await create_telegraph_page("Test Page", "# Test Content")

            self.assertEqual(result, 'https://telegra.ph/Test-Page-01-01')

            mock_session.post.assert_awaited_once()
            args, kwargs = mock_session.post.call_args
            self.assertEqual(args[0], 'https://api.telegra.ph/createPage')
            self.assertIn('data', kwargs)
            self.assertIn('content', kwargs['data'])

            try:
                content_nodes = json.loads(kwargs['data']['content'])
                self.assertIsInstance(content_nodes, list)
            except json.JSONDecodeError:
                self.fail("Content was not a valid JSON string of nodes")

    @pytest.mark.asyncio
    async def test_create_telegraph_page_failure(self):
        """Test failure handling when creating a Telegraph page."""
        # Mock the shared HTTP session to simulate failure
        with patch('bot.handlers.content.get_http_session', new_callable=AsyncMock) as mock_get_session:
            mock_session = AsyncMock()
            mock_get_session.return_value = mock_session

            mock_response = AsyncMock()
            mock_response.json = AsyncMock(return_value={
                'ok': False,
                'error': 'SOME_ERROR'
            })
            mock_response.raise_for_status = MagicMock()

            mock_ctx = AsyncMock()
            mock_ctx.__aenter__.return_value = mock_response
            mock_ctx.__aexit__.return_value = False

            mock_session.post = AsyncMock(return_value=mock_ctx)

            result = await create_telegraph_page("Test Page", "# Test Content")

            self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main() 
