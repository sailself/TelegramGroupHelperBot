"""Test Telegraph functionality for handling long messages."""

import json
import unittest
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from bs4 import BeautifulSoup

from bot.handlers import create_telegraph_page, markdown_to_telegraph_nodes, html_to_telegraph_nodes


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
        # Mock the requests.post method
        with patch('requests.post') as mock_post:
            # Configure the mock to return a successful response
            mock_response = MagicMock()
            mock_response.json.return_value = {
                'ok': True,
                'result': {
                    'url': 'https://telegra.ph/Test-Page-01-01'
                }
            }
            mock_post.return_value = mock_response
            
            # Call the function
            result = await create_telegraph_page("Test Page", "# Test Content")
            
            # Assert the function returned the expected URL
            self.assertEqual(result, 'https://telegra.ph/Test-Page-01-01')
            
            # Verify the mock was called correctly
            mock_post.assert_called_once()
            args, kwargs = mock_post.call_args
            
            # Check that the URL is correct
            self.assertEqual(args[0], 'https://api.telegra.ph/createPage')
            
            # Check that the content was converted to nodes and included in the request
            self.assertIn('data', kwargs)
            self.assertIn('content', kwargs['data'])
            
            # Content should be a JSON string of nodes
            try:
                content_nodes = json.loads(kwargs['data']['content'])
                self.assertIsInstance(content_nodes, list)
            except json.JSONDecodeError:
                self.fail("Content was not a valid JSON string of nodes")

    @pytest.mark.asyncio
    async def test_create_telegraph_page_failure(self):
        """Test failure handling when creating a Telegraph page."""
        # Mock the requests.post method to simulate failure
        with patch('requests.post') as mock_post:
            # Configure the mock to return a failed response
            mock_response = MagicMock()
            mock_response.json.return_value = {
                'ok': False,
                'error': 'SOME_ERROR'
            }
            mock_post.return_value = mock_response
            
            # Call the function
            result = await create_telegraph_page("Test Page", "# Test Content")
            
            # Assert the function returned None on failure
            self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main() 